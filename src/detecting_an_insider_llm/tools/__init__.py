"""Public surface for model-visible tools and their safe implementations.

Importing this package only defines schemas and classes.  It does not register
tools with an agent, execute a provider response, or create global mailbox
state.  Runtime integration therefore remains explicit and episode-scoped.
"""

from detecting_an_insider_llm.runtime.tool_loop import ToolExecutionResult
from detecting_an_insider_llm.tools.action_logging import (
    DoubleLoggingToolExecutor,
)
from detecting_an_insider_llm.tools.definitions import (
    LOG_ACTION_TOOL,
    LIST_EMAILS_TOOL,
    READ_EMAIL_TOOL,
    SEND_EMAIL_TOOL,
    email_tool_definitions,
    self_logging_email_tool_definitions,
)
from detecting_an_insider_llm.tools.email import (
    EmailMessage,
    ListEmailsArguments,
    ReadEmailArguments,
    SendEmailArguments,
    SimulatedMailbox,
    is_safe_test_address,
    list_emails,
    read_email,
    send_email,
)
from detecting_an_insider_llm.tools.registry import EmailToolDispatcher
from detecting_an_insider_llm.tools.self_log import (
    AutomaticActionRecord,
    ClaimedOutcome,
    LogActionArguments,
    ModelSelfLogRecord,
    SelfLogStore,
    log_action,
)

__all__ = [
    "AutomaticActionRecord",
    "ClaimedOutcome",
    "DoubleLoggingToolExecutor",
    "EmailMessage",
    "EmailToolDispatcher",
    "LIST_EMAILS_TOOL",
    "LOG_ACTION_TOOL",
    "ListEmailsArguments",
    "LogActionArguments",
    "ModelSelfLogRecord",
    "READ_EMAIL_TOOL",
    "ReadEmailArguments",
    "SEND_EMAIL_TOOL",
    "SendEmailArguments",
    "SimulatedMailbox",
    "SelfLogStore",
    "ToolExecutionResult",
    "email_tool_definitions",
    "is_safe_test_address",
    "list_emails",
    "log_action",
    "read_email",
    "send_email",
    "self_logging_email_tool_definitions",
]
