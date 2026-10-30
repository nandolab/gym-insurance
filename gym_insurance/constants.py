from pathlib import Path
import logging

MODELSPATH = Path("./models")
DATAPATH = Path("C:\\Repos\\gym-insurance\\experiment_results\\data")
#BUDGET_LIST = [1_000_000] #, 5_000_000, 10_000_000, 15_000_000]
#BUDGET_LIST = [15_000_000 , 20_000_000, 25_000_000, 30_000_000]
#BUDGET_LIST = [ 10_000_000, 20_000_000, 30_000_000]
BUDGET_LIST = [ 30_000_000]


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


configure_logging()
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
