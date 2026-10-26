import uuid

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile, status

from app.core.exception import PayloadTooLargeError
from app.core.rate_limiting import rate_limit
from app.dependencies import get_current_user, get_document_service
from app.models.db.user import User
from app.models.document_model import (
    ChunkListResponse,
    ChunkResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentTextCreateRequest,
)
from app.models.response_model import GlobalResponse, SuccessResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])

read_limit = [Depends(rate_limit("documents:read"))]
write_limit = [Depends(rate_limit("documents:write"))]


@router.post(
    "",
    response_model=GlobalResponse[DocumentResponse],
    status_code=status.HTTP_201_CREATED,
    dependencies=write_limit,
)
async def upload_document(
    request: Request,
    file: UploadFile = File(description="PDF, Markdown (.md) or plain text (.txt)"),
    user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> SuccessResponse[DocumentResponse]:
    """Upload a file. It is chunked and embedded in the background; poll until status is ready."""
    max_bytes: int = request.app.state.settings.MAX_UPLOAD_BYTES
    raw = await file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise PayloadTooLargeError(f"Files must be at most {max_bytes // (1024 * 1024)} MB")
    document = await service.create_from_upload(
        user.id, file.filename or "upload.txt", file.content_type, raw
    )
    return SuccessResponse(
        status=201, message="Document queued", data=DocumentResponse.model_validate(document)
    )


@router.post(
    "/text",
    response_model=GlobalResponse[DocumentResponse],
    status_code=status.HTTP_201_CREATED,
    dependencies=write_limit,
)
async def create_text_document(
    body: DocumentTextCreateRequest,
    user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> SuccessResponse[DocumentResponse]:
    document = await service.create_from_text(user.id, body.title, body.content)
    return SuccessResponse(
        status=201, message="Document queued", data=DocumentResponse.model_validate(document)
    )


@router.get("", response_model=GlobalResponse[DocumentListResponse], dependencies=read_limit)
async def list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> SuccessResponse[DocumentListResponse]:
    items, total = await service.list_documents(user.id, limit=limit, offset=offset)
    page = DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in items],
        total=total,
        limit=limit,
        offset=offset,
    )
    return SuccessResponse(data=page)


@router.get(
    "/{document_id}", response_model=GlobalResponse[DocumentResponse], dependencies=read_limit
)
async def get_document(
    document_id: uuid.UUID,
    user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> SuccessResponse[DocumentResponse]:
    document = await service.get(user.id, document_id)
    return SuccessResponse(data=DocumentResponse.model_validate(document))


@router.get(
    "/{document_id}/chunks",
    response_model=GlobalResponse[ChunkListResponse],
    dependencies=read_limit,
)
async def list_chunks(
    document_id: uuid.UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> SuccessResponse[ChunkListResponse]:
    items, total = await service.list_chunks(user.id, document_id, limit=limit, offset=offset)
    page = ChunkListResponse(
        items=[ChunkResponse.model_validate(c) for c in items],
        total=total,
        limit=limit,
        offset=offset,
    )
    return SuccessResponse(data=page)


@router.post(
    "/{document_id}/reindex",
    response_model=GlobalResponse[DocumentResponse],
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=write_limit,
)
async def reindex_document(
    document_id: uuid.UUID,
    user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> SuccessResponse[DocumentResponse]:
    document = await service.reindex(user.id, document_id)
    return SuccessResponse(
        status=202, message="Re-index queued", data=DocumentResponse.model_validate(document)
    )


@router.delete("/{document_id}", response_model=GlobalResponse[None], dependencies=write_limit)
async def delete_document(
    document_id: uuid.UUID,
    user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> SuccessResponse[None]:
    await service.delete(user.id, document_id)
    return SuccessResponse(message="Document deleted")
