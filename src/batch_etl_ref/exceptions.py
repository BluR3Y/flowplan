
class EtlError(Exception):
    """Base exception for the ETL library."""
    pass

class ConfigError(EtlError):
    """Configuration validation or loading error."""
    pass

class SourceError(EtlError):
    """Error during data ingestion."""
    pass

class TransformError(EtlError):
    """Error during data transformation."""
    pass

class ExportError(EtlError):
    """Error during data export."""
    pass

class ExprError(EtlError):
    """Error in expression evaluation."""
    pass