# Custom exception class

# Importing the sys module to get system-specific information (used for error traceback)
import sys

# Define a function that helps generate a detailed error message
def error_message_detail(error, error_detail: sys):
    # This extracts exception info including traceback (tb), type, and value
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Get the exact line number where the error occurred
    line_number = exc_tb.tb_lineno

    # Format the error message with file name, line number, and the actual error message
    error_message = f"Error occurred in Python script [{file_name}] at line [{line_number}] error message: [{str(error)}]"

    # Return the formatted error message string
    return error_message

# Define a custom exception class that inherits from the built-in Exception class
class CustomException(Exception):
    # Constructor takes two parameters:
    # error_message → the original error (e.g., 'division by zero')
    # error_detail → the sys module, used to extract traceback info
    def __init__(self, error_message, error_detail: sys):
        # Call the parent class constructor with the error message
        super().__init__(error_message) 

        # Use the error_message_detail function to generate a formatted message with filename and line number
        self.error_message = error_message_detail(error_message, error_detail)

    # This method tells Python what to show when we print the exception
    def __str__(self):
        # Return the formatted custom error message instead of default error string
        return self.error_message




