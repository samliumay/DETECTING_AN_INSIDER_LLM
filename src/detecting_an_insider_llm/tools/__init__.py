"""Public surface for model-visible tools and their safe implementations.

Importing this package only defines schemas and classes.  It does not register
tools with an agent, execute a provider response, or create global mailbox
state.  Runtime integration will therefore remain an explicit later step.
"""

from detecting_an_insider_llm.tools.definitions import (
    READ_EMAIL_TOOL,
    SEND_EMAIL_TOOL,
    email_tool_definitions,
)
from detecting_an_insider_llm.tools.email import (
    EmailMessage,
    ReadEmailArguments,
    SendEmailArguments,
    SimulatedMailbox,
    ToolExecutionResult,
    is_safe_test_address,
    read_email,
    send_email,
)

__all__ = [
    "READ_EMAIL_TOOL",
    "SEND_EMAIL_TOOL",
    "EmailMessage",
    "ReadEmailArguments",
    "SendEmailArguments",
    "SimulatedMailbox",
    "ToolExecutionResult",
    "email_tool_definitions",
    "is_safe_test_address",
    "read_email",
    "send_email",
]
