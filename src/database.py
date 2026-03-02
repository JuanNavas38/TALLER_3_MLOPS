import logging
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER

logger = logging.getLogger(__name__)


def get_engine() -> Engine:
    """Crea y retorna el engine de conexión a MySQL."""
    url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url, pool_pre_ping=True)
    logger.info("Conexión a MySQL establecida: %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)
    return engine


def create_tables(engine: Engine) -> None:
    """Crea las tablas raw y procesada si no existen."""
    with engine.connect() as conn:
        # Tabla datos raw (sin preprocesamiento)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS penguins_raw (
                id INT AUTO_INCREMENT PRIMARY KEY,
                species VARCHAR(50),
                island VARCHAR(50),
                bill_length_mm FLOAT,
                bill_depth_mm FLOAT,
                flipper_length_mm FLOAT,
                body_mass_g FLOAT,
                sex VARCHAR(10),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # Tabla datos preprocesados (listos para entrenar)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS penguins_processed (
                id INT AUTO_INCREMENT PRIMARY KEY,
                species INT,
                island INT,
                bill_length_mm FLOAT,
                bill_depth_mm FLOAT,
                flipper_length_mm FLOAT,
                body_mass_g FLOAT,
                sex INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
    logger.info("Tablas creadas correctamente.")


def clear_tables(engine: Engine) -> None:
    """Borra el contenido de las tablas."""
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM penguins_processed"))
        conn.execute(text("DELETE FROM penguins_raw"))
        conn.commit()
    logger.info("Tablas limpiadas correctamente.")


def load_raw_data(engine: Engine, df: pd.DataFrame) -> None:
    """Inserta datos raw en la base de datos."""
    df.to_sql(
        name="penguins_raw",
        con=engine,
        if_exists="append",
        index=False,
    )
    logger.info("Datos raw insertados: %d filas.", len(df))


def load_processed_data(engine: Engine, df: pd.DataFrame) -> None:
    """Inserta datos preprocesados en la base de datos."""
    df.to_sql(
        name="penguins_processed",
        con=engine,
        if_exists="append",
        index=False,
    )
    logger.info("Datos procesados insertados: %d filas.", len(df))


def read_processed_data(engine: Engine) -> pd.DataFrame:
    """Lee los datos preprocesados para entrenamiento."""
    df = pd.read_sql("SELECT * FROM penguins_processed", con=engine)
    df = df.drop(columns=["id", "created_at"])
    logger.info("Datos procesados leídos: %d filas.", len(df))
    return df