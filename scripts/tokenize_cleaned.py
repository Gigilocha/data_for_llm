import json
from transformers import PreTrainedTokenizerFast
from common.config import get_config, get_env_settings
from common.logger import get_logger
from common.paths import tokenizer_dir
from tokenizer.tokenize import run_tokenization

logger = get_logger(__name__)

def main():
    config = get_config()
    env = get_env_settings()
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir(env.output_dir))
    manifest = run_tokenization(tokenizer, env.output_dir, config)
    
    manifest_path = env.output_dir / "tokenized" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Готово! Манифест записан в %s", manifest_path)