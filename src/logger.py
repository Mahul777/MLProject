# ✅ Logging setup

# Importing required modules
import logging  # Provides logging functionalities to track events and errors
import os       # Helps with file and directory operations (like creating folders)
from datetime import datetime  # Used to fetch current timestamp for unique log filenames

# 🕒 Create a log file name using current date and time (e.g., 06_12_2025_15_30_45.log)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 📂 Build path to save the log file → <current_directory>/logs/<timestamp>.log
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# 📁 Create the 'logs' folder if it doesn't exist already
os.makedirs(logs_path, exist_ok=True)

# 📝 Final full path of the log file (technically redundant since logs_path already includes file name)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# ⚙️ Configure the logging behavior using basicConfig
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Where the logs will be saved (file path)
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Log message format:
    # Example log: [2025-06-12 15:30:45] 23 root - INFO - Logging has Started
    level=logging.INFO  # Minimum log level to record (INFO and above)
)


