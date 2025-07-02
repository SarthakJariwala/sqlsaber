"""Exception classes for LLM client errors."""

from typing import Any


class LLMClientError(Exception):
    """Base exception for LLM client errors."""

    def __init__(
        self,
        message: str,
        error_type: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.request_id = request_id


class AuthenticationError(LLMClientError):
    """Authentication failed - invalid API key."""

    def __init__(self, message: str = "Invalid API key", **kwargs):
        super().__init__(message, "authentication_error", **kwargs)


class PermissionError(LLMClientError):
    """Permission denied for the requested resource."""

    def __init__(self, message: str = "Permission denied", **kwargs):
        super().__init__(message, "permission_error", **kwargs)


class NotFoundError(LLMClientError):
    """Requested resource not found."""

    def __init__(self, message: str = "Resource not found", **kwargs):
        super().__init__(message, "not_found_error", **kwargs)


class InvalidRequestError(LLMClientError):
    """Invalid request format or content."""

    def __init__(self, message: str = "Invalid request", **kwargs):
        super().__init__(message, "invalid_request_error", **kwargs)


class RequestTooLargeError(LLMClientError):
    """Request exceeds maximum allowed size."""

    def __init__(self, message: str = "Request too large", **kwargs):
        super().__init__(message, "request_too_large", **kwargs)


class RateLimitError(LLMClientError):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(message, "rate_limit_error", **kwargs)


class APIError(LLMClientError):
    """Internal API error."""

    def __init__(self, message: str = "Internal API error", **kwargs):
        super().__init__(message, "api_error", **kwargs)


class OverloadedError(LLMClientError):
    """API is temporarily overloaded."""

    def __init__(self, message: str = "API temporarily overloaded", **kwargs):
        super().__init__(message, "overloaded_error", **kwargs)


# Mapping of HTTP status codes to exception classes
STATUS_CODE_TO_EXCEPTION = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: PermissionError,
    404: NotFoundError,
    413: RequestTooLargeError,
    429: RateLimitError,
    500: APIError,
    529: OverloadedError,
}


def create_exception_from_response(
    status_code: int,
    response_data: dict[str, Any],
    request_id: str | None = None,
) -> LLMClientError:
    """Create appropriate exception from HTTP response."""
    error_data = response_data.get("error", {})
    message = error_data.get("message", f"HTTP {status_code} error")
    error_type = error_data.get("type")

    exception_class = STATUS_CODE_TO_EXCEPTION.get(status_code, LLMClientError)

    # Handle base vs subclass constructors
    if exception_class == LLMClientError:
        return exception_class(
            message,
            error_type,
            status_code,
            request_id,
        )
    else:
        # Subclasses only take message and **kwargs
        return exception_class(
            message,
            status_code=status_code,
            request_id=request_id,
        )
