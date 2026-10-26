from fastapi import APIRouter, Depends

from app.core.rate_limiting import rate_limit
from app.dependencies import get_current_user, get_qa_service, get_retrieval_service
from app.models.db.user import User
from app.models.qa_model import AskRequest, AskResponse, SearchRequest, SearchResponse
from app.models.response_model import GlobalResponse, SuccessResponse
from app.services.qa_service import QAService
from app.services.retrieval_service import RetrievalService

router = APIRouter(tags=["Question answering"])


@router.post(
    "/search",
    response_model=GlobalResponse[SearchResponse],
    dependencies=[Depends(rate_limit("search"))],
)
async def search(
    body: SearchRequest,
    user: User = Depends(get_current_user),
    service: RetrievalService = Depends(get_retrieval_service),
) -> SuccessResponse[SearchResponse]:
    """Retrieval only, no LLM call: the chunks /ask would answer from, with their scores."""
    result = await service.search(
        user.id,
        body.query,
        top_k=body.top_k,
        mode=body.mode,
        rerank=body.rerank,
        document_ids=body.document_ids,
    )
    return SuccessResponse(data=SearchResponse(hits=result.hits, timings_ms=result.timings_ms))


@router.post(
    "/ask",
    response_model=GlobalResponse[AskResponse],
    dependencies=[Depends(rate_limit("ask"))],
)
async def ask(
    body: AskRequest,
    user: User = Depends(get_current_user),
    service: QAService = Depends(get_qa_service),
) -> SuccessResponse[AskResponse]:
    """Answer from the user's documents, citing the chunks used as [n] markers."""
    answer = await service.ask(
        user.id, body.question, top_k=body.top_k, document_ids=body.document_ids
    )
    return SuccessResponse(data=answer)
