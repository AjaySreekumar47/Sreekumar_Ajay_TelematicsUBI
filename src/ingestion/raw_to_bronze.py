# /content/src/ingestion/raw_to_bronze.py
import os, glob, logging
import pandas as pd

RAW_DIR = "/content/Sreekumar_Ajay_TelematicsUBI/data/raw/telematics_full"
BRONZE_DIR = "/content/Sreekumar_Ajay_TelematicsUBI/data/bronze"
BATCH_SIZE = 50000  # adjust depending on dataset size

os.makedirs(BRONZE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("raw_to_bronze")

def raw_to_bronze():
    files = glob.glob(os.path.join(RAW_DIR, "*.parquet"))
    if not files:
        logger.error("No raw parquet files found in %s", RAW_DIR)
        return

    bronze_idx = 0
    for f in files:
        df = pd.read_parquet(f)
        logger.info("Loaded %s rows from %s", len(df), f)

        # Deduplicate
        before = len(df)
        df = df.drop_duplicates(subset=["event_id"])
        after = len(df)
        if after < before:
            logger.info("Deduplicated: %s → %s", before, after)

        # Write in Bronze-sized chunks
        for i in range(0, len(df), BATCH_SIZE):
            chunk = df.iloc[i:i+BATCH_SIZE]
            out_file = os.path.join(BRONZE_DIR, f"bronze_batch{bronze_idx}.parquet")
            chunk.to_parquet(out_file, index=False, engine="pyarrow")
            logger.info("Wrote %s rows → %s", len(chunk), out_file)
            bronze_idx += 1

    logger.info("✅ Completed raw → bronze. Files written: %s", bronze_idx)
    logger.info("Raw file %s (%s rows) → Bronze file %s (%s rows)", f, before, out_file, len(chunk))

if __name__ == "__main__":
    raw_to_bronze()
