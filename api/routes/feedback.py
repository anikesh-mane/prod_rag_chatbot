"""Feedback endpoint for collecting user ratings."""

from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request, status

from api.dependencies import OptionalUserDep, SettingsDep
from monitoring.metrics import record_feedback
from schemas import (
    ErrorCode,
    ErrorDetail,
    FeedbackRequest,
    FeedbackResponse,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
)
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    settings: SettingsDep,
    current_user: OptionalUserDep,
) -> FeedbackResponse:
    """Submit feedback for a chat response.

    Feedback is stored for:
    - Quality monitoring
    - Training data generation
    - Continuous improvement
    """
    feedback_id = uuid4()

    logger.info(
        "Feedback received",
        feedback_id=str(feedback_id),
        query_id=body.query_id,
        rating=body.rating,
        feedback_type=body.feedback_type,
        user_id=current_user.user_id if current_user else "anonymous",
    )

    try:
        # TODO: Store feedback in database
        # feedback_record = FeedbackRecord(
        #     feedback_id=feedback_id,
        #     query_id=UUID(body.query_id),
        #     user_id=UUID(current_user.user_id) if current_user else None,
        #     rating=body.rating,
        #     comment=body.comment,
        #     feedback_type=body.feedback_type,
        # )
        # await feedback_repository.save(feedback_record)

        # Record metrics
        record_feedback(rating=body.rating, feedback_type=body.feedback_type)

        logger.info(
            "Feedback stored",
            feedback_id=str(feedback_id),
            rating=body.rating,
        )

        return FeedbackResponse(
            feedback_id=feedback_id,
            status="accepted",
            message="Thank you for your feedback",
        )

    except Exception as e:
        logger.error(
            "Failed to store feedback",
            feedback_id=str(feedback_id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message="Failed to store feedback",
                request_id=getattr(request.state, "request_id", feedback_id),
            ).model_dump(),
        )


@router.get("/feedback/stats")
async def get_feedback_stats(
    request: Request,
    settings: SettingsDep,
    current_user: OptionalUserDep,
) -> dict:
    """Get feedback statistics (admin only in production)."""
    # TODO: Implement actual stats from database
    return {
        "total_feedback": 0,
        "average_rating": 0.0,
        "rating_distribution": {
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "5": 0,
        },
        "feedback_types": {
            "rating": 0,
            "correction": 0,
            "flag": 0,
        },
    }
