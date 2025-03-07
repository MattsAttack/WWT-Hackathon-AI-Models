"""Taken from <https://github.com/encode/starlette/discussions/2363#discussioncomment-7802495>."""

from collections.abc import AsyncIterable, Iterable

from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import JSONResponse, StreamingResponse


class JSONStreamingResponse(StreamingResponse, JSONResponse):
    """StreamingResponse, but it's JSON array."""

    def __init__(
        self,
        content: Iterable | AsyncIterable,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
        background: BackgroundTask | None = None,
    ) -> None:
        if isinstance(content, AsyncIterable):
            self._content_iterable: AsyncIterable = content
        else:
            self._content_iterable = iterate_in_threadpool(content)

        async def body_iterator() -> AsyncIterable[bytes]:
            yield b"["

            first_item = True

            async for content_ in self._content_iterable:
                if isinstance(content_, BaseModel):
                    content_ = content_.model_dump()  # noqa: PLW2901

                # Add comma before all items except the first
                if not first_item:
                    yield b","
                else:
                    first_item = False

                yield self.render(content_)

            # End with closing bracket
            yield b"]"

        self.body_iterator = body_iterator()
        self.status_code = status_code
        if media_type is not None:
            self.media_type = media_type
        self.background = background
        self.init_headers(headers)
