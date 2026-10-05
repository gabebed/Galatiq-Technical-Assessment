"""Concrete tools, grouped into per-agent toolsets.

* Validation agent: ``lookup_inventory``, ``check_stock`` (read-only inventory).
* Payment agent: ``mock_payment``, bound to a single ``PaymentAuthorization``.

Neither toolset exposes SQL, Python execution, or the other agent's tools.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field

from invoice_processor.inventory import InventoryLookup
from invoice_processor.models import PaymentResult, PaymentStatus
from invoice_processor.payment import (
    PaymentAuthorization,
    PaymentFunction,
    execute_authorized_payment,
    mock_payment,
)
from invoice_processor.tools import Tool, ToolArgs, ToolRefused, Toolset

# --------------------------------------------------------------------------- #
# Validation agent: inventory tools
# --------------------------------------------------------------------------- #


class ItemArgs(ToolArgs):
    item: str = Field(min_length=1, max_length=100, description="Exact inventory item name, e.g. 'WidgetA'.")


class StockCheckArgs(ItemArgs):
    quantity: int = Field(gt=0, le=1_000_000, description="Total quantity requested on the invoice.")


class InventoryRecord(BaseModel):
    item: str
    found: bool
    stock: int | None = None


class StockCheck(BaseModel):
    item: str
    found: bool
    requested: int
    available: int | None
    sufficient: bool
    explanation: str


def build_validation_tools(inventory: InventoryLookup) -> Toolset:
    """Read-only inventory tools. Item names are passed as query parameters, never as SQL."""

    def lookup_inventory(args: ItemArgs) -> InventoryRecord:
        stock = inventory.get_stock_levels([args.item]).get(args.item)
        return InventoryRecord(item=args.item, found=stock is not None, stock=stock)

    def check_stock(args: StockCheckArgs) -> StockCheck:
        stock = inventory.get_stock_levels([args.item]).get(args.item)
        if stock is None:
            explanation = f"{args.item} is not in inventory."
        elif stock == 0:
            explanation = f"{args.item} has zero stock."
        elif args.quantity > stock:
            explanation = f"Requested {args.quantity} exceeds available stock of {stock}."
        else:
            explanation = f"Requested {args.quantity} is within available stock of {stock}."
        return StockCheck(
            item=args.item, found=stock is not None, requested=args.quantity, available=stock,
            sufficient=stock is not None and args.quantity <= stock, explanation=explanation,
        )

    return Toolset("validation", [
        Tool("lookup_inventory", "Look up an inventory item and its current stock level.", ItemArgs, lookup_inventory),
        Tool("check_stock", "Check whether a quantity of an item is available in stock.", StockCheckArgs, check_stock),
    ])


# --------------------------------------------------------------------------- #
# Payment agent: mock_payment bound to one authorization
# --------------------------------------------------------------------------- #


class PaymentArgs(ToolArgs):
    vendor: str = Field(min_length=1, max_length=200, description="Vendor name exactly as approved.")
    amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2, description="Amount exactly as approved.")


class PaymentToolset(Toolset):
    """Payment toolset that also records the payments actually executed."""

    def __init__(self, authorization: PaymentAuthorization, pay: PaymentFunction) -> None:
        self.authorization = authorization
        self.payments: list[PaymentResult] = []

        def mock_payment_tool(args: PaymentArgs) -> PaymentResult:
            auth = self.authorization
            if any(p.status is PaymentStatus.PAID for p in self.payments):
                raise ToolRefused(f"Invoice {auth.invoice_number} has already been paid; duplicate payments are not allowed.")
            if args.vendor != auth.vendor or args.amount != auth.amount:
                raise ToolRefused(
                    f"Requested payment ({args.amount} to {args.vendor!r}) does not match the approved payment "
                    f"({auth.amount} {auth.currency} to {auth.vendor!r}). Only the approved payment can be made."
                )
            # Execute with the authorized values, not the agent's copies.
            result = execute_authorized_payment(auth, pay)
            self.payments.append(result)
            return result

        super().__init__("payment", [
            Tool("mock_payment", "Pay the approved invoice. Vendor and amount must match the approval exactly.",
                 PaymentArgs, mock_payment_tool),
        ], max_calls=3)

    @property
    def completed_payment(self) -> PaymentResult | None:
        return next((p for p in self.payments if p.status is PaymentStatus.PAID), None)


def build_payment_tools(authorization: PaymentAuthorization, pay: PaymentFunction = mock_payment) -> PaymentToolset:
    """The payment tool exists only for an authorization from ``authorize_payment``."""
    if not isinstance(authorization, PaymentAuthorization):
        raise TypeError("build_payment_tools requires a PaymentAuthorization from authorize_payment()")
    return PaymentToolset(authorization, pay)
