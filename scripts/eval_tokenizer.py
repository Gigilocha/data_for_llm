import sys
import io
import json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from common.config import get_config, get_env_settings
from common.logger import get_logger
from common.paths import tokenizer_dir, tokenizer_sample_dir
from tokenizer.evaluate import evaluate_tokenizer

logger = get_logger(__name__)

def main():
    config = get_config()
    env = get_env_settings()
    
    tok_dir = tokenizer_dir(env.output_dir)
    sample_path = tokenizer_sample_dir(env.output_dir) / "sample.parquet"
    
    if not tok_dir.exists():
        raise FileNotFoundError(f"Токенизатор не найден в {tok_dir}. Сначала обучите его.")
    if not sample_path.exists():
        raise FileNotFoundError(f"Срез не найден в {sample_path}. Сначала соберите срез.")
        
    logger.info("Загрузка токенизатора из %s...", tok_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir)
    
    logger.info("Загрузка тестовых текстов из %s...", sample_path)
    ds = load_dataset("parquet", data_files=str(sample_path), split="train")
    # Берем первые 100 текстов для оценки (чтобы не гонять всё)
    test_texts = [row["text"] for row in ds.take(100)]
    
    logger.info("Запуск оценки...")
    metrics = evaluate_tokenizer(tokenizer, test_texts)
    
    print("\n=== ОЦЕНКА ТОКЕНИЗАТОРА ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("===========================\n")
    
    # Сохраняем результат в отдельный файл
    out_path = tok_dir / "evaluation.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Метрики сохранены в %s", out_path)

if __name__ == "__main__":
    main()