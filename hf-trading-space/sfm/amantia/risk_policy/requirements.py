from __future__ import annotations

TRADING_ACTIONS = {
    "connect_brokerage_api",
    "place_market_order",
    "place_limit_order",
    "cancel_order",
    "modify_order",
    "close_position",
    "open_margin_position",
    "set_stop_loss",
    "change_trading_risk_limits",
    "rebalance_portfolio",
}

FINANCE_ACTIONS = {
    "view_account_balance",
    "access_financial_data",
    "initiate_bank_transfer",
    "withdraw_funds",
    "approve_invoice_payment",
    "charge_customer",
    "refund_payment",
    "issue_payout",
}

VALUE_TRANSFER_ACTIONS = {
    "initiate_bank_transfer",
    "withdraw_funds",
    "approve_invoice_payment",
    "charge_customer",
    "refund_payment",
    "issue_payout",
}

DESTRUCTIVE_ACTIONS = {
    "delete_resource",
    "delete_file",
    "erase_memory",
    "drop_database",
    "delete_database",
}

CODE_OR_OPS_ACTIONS = {
    "run_shell_command",
    "execute_code",
    "modify_database",
    "deploy_code",
    "install_package",
    "deploy_config_change",
}

EXTERNAL_COMM_ACTIONS = {
    "send_email_external",
    "share_file_external",
    "send_email",
}

HIGH_RISK_DOMAINS = {"finance", "trading", "database", "ops", "security"}
