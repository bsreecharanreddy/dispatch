from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GenerateRequest(_message.Message):
    __slots__ = ("request_id", "prompt", "max_new_tokens")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    prompt: str
    max_new_tokens: int
    def __init__(self, request_id: _Optional[str] = ..., prompt: _Optional[str] = ..., max_new_tokens: _Optional[int] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("request_id", "text", "is_final", "t_emit_unix")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    T_EMIT_UNIX_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    text: str
    is_final: bool
    t_emit_unix: float
    def __init__(self, request_id: _Optional[str] = ..., text: _Optional[str] = ..., is_final: _Optional[bool] = ..., t_emit_unix: _Optional[float] = ...) -> None: ...
