from dataset.sampling.budget import SamplingBudget, compute_budget, split_budget_by_weights


def test_compute_budget():
    langs = {
        "ru": {"share": 0.6, "chars_per_token": 3.0},
        "en": {"share": 0.4, "chars_per_token": 3.0},
    }
    b = compute_budget(target_tokens=1_000_000, overshoot=2.0, languages=langs)
    assert b.chars_per_lang["ru"] == 3_600_000
    assert b.chars_per_lang["en"] == 2_400_000
    assert b.total_chars == 6_000_000
    assert b.tokenizer_sample_chars == 60_000   # 1% от 6M

def test_split_budget():
    assert split_budget_by_weights(100, [0.7, 0.3]) == [70, 30]
    assert split_budget_by_weights(100, [0.5, 0.5]) == [50, 50]
    assert split_budget_by_weights(10, [1/3, 1/3, 1/3]) in (
        [3, 3, 4], [3, 4, 3], [4, 3, 3]
    )   # сумма = 10, распределение по остатку
    assert sum(split_budget_by_weights(21_600_000, [0.7, 0.3])) == 21_600_000
