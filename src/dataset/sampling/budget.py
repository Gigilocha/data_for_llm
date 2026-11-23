from dataclasses import dataclass


# Класс описывающий ограничения на количество генерируемых текстовых фрагментов
@dataclass(frozen=True)
class SamplingBudget:
    chars_per_lang: dict[str, int]
    total_chars: int
    tokenizer_sample_chars: int


# Функция для вычисления бюджета 
def compute_budget(target_tokens: int, overshoot: float, languages: dict) -> SamplingBudget:
    chars = {}
    for lang, cfg in languages.items():
        chars[lang] = int(target_tokens * cfg["share"] * cfg["chars_per_token"] * overshoot)

    total = sum(chars.values())
    sample_chars = min(int(total * 0.01), 1_000_000)

    return SamplingBudget(
        chars_per_lang=chars,
        total_chars=total,
        tokenizer_sample_chars=sample_chars,
    )

# 
def split_budget_by_weights(total: int, weights: list[float]) -> list[int]:
    if not weights:
        return []
    if len(weights) == 1:
        return [total]

    w_sum = sum(weights)
    if w_sum == 0:
        raise ValueError("Сумма весов не может быть равна нулю")
    if any(w < 0 for w in weights):
        raise ValueError("Веса не могут быть отрицательными")

    ideal = [total * w / w_sum for w in weights]
    floors = [int(x) for x in ideal]

    remainder = total - sum(floors)
    if remainder == 0:
        return floors

    fracs = [(ideal[i] - floors[i], i) for i in range(len(weights))]
    fracs.sort(reverse=True)

    for k in range(remainder):
        _, idx = fracs[k]
        floors[idx] += 1

    return floors