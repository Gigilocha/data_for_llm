from transformers import PreTrainedTokenizerFast
from typing import List, Dict

def evaluate_tokenizer(tokenizer: PreTrainedTokenizerFast, test_texts: List[str]) -> Dict:
    """
    Считает метрики качества токенизатора:
    - compression_ratio: сколько символов на один токен (чем больше, тем лучше)
    - tokens_per_word: сколько токенов на одно слово (идеал ~1.2-1.5)
    - roundtrip_success: сохраняется ли текст без потерь
    """
    
    total_chars = 0
    total_tokens = 0
    total_words = 0
    roundtrip_ok = 0

    # Берем только непустые тексты
    valid_texts = [t for t in test_texts if t and t.strip()]
    if not valid_texts:
        return {"error": "Нет текстов для оценки"}

    for text in valid_texts:
        # 1. Кодируем
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens_count = len(ids)
        
        # 2. Декодируем обратно
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        
        # 3. Считаем статистику
        total_chars += len(text)
        total_tokens += tokens_count
        total_words += len(text.split())
        
        # Сравниваем (игнорируем мелкие артефакты пробелов на краях)
        if decoded.strip() == text.strip():
            roundtrip_ok += 1

    # Считаем средние значения
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    tokens_per_word = total_tokens / total_words if total_words > 0 else 0
    roundtrip_rate = roundtrip_ok / len(valid_texts)

    return {
        "compression_ratio": round(compression_ratio, 2),
        "tokens_per_word": round(tokens_per_word, 2),
        "roundtrip_success_rate": round(roundtrip_rate, 2),
        "total_texts_evaluated": len(valid_texts),
    }