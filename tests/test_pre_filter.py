import pytest
from src.filters.pre_filter import PreFilterResult, pre_filter_job


VALID_DESC = (
    "We are seeking an experienced Executive and Leadership Coach to join our talent team. "
    "You will coach senior executives, manage organizational culture initiatives, and facilitate leadership development. "
    "Requires 5+ years of corporate coaching experience in European or global markets."
)


@pytest.mark.parametrize("title", [
    # English athletic / sports / vocal / gaming / pet
    "Swimming Coach",
    "Head Tennis Coach",
    "Fitness Instructor / Personal Trainer",
    "CrossFit Coach & Gym Trainer",
    "Cross-Fit Instructor",
    "Vocal & Drama Coach",
    "Esports Valorant Coach",
    "E-sports Coach",
    "Dog Agility Coach / Puppy Trainer",
    "Martial Arts Boxing & Karate Coach",
    "Ski & Snowboard Instructor",
    "Yoga & Pilates Teacher",
    "Sports Coach",
    "Athletic Coach",
    "Animal Coach",
    # Polish athletic / sports / vocal / pet
    "Trener pływania dla dzieci",
    "Trener piłki nożnej (UEFA A)",
    "Trener Personalny / Instruktor Siłowni",
    "Koszykówka Trener Młodzieży",
    "Treser psów / Trener zwierząt",
    "Instruktor tańca i choreografii",
    "Instruktor jeździectwa",
    "Trener wspinaczki sportowej",
    "Trener sportowy",
    # Ukrainian athletic / sports / vocal / pet
    "Тренер з плавання для підлітків",
    "Тренер з тенісу",
    "Фітнес-тренер / Інструктор тренажерного залу",
    "Тренер з футболу",
    "Вокальний тренер / Викладач вокалу",
    "Тренер з гімнастики",
    "Тренер з боксу та карате",
    "Тренер собак / Кінолог",
    "Спортивний тренер",
])
def test_pre_filter_rejects_sports_and_vocal_titles(title: str):
    res = pre_filter_job(title, VALID_DESC)
    assert res.passed is False
    assert res.reason is not None
    assert "EXCLUDED_TITLE_REGEX" in res.reason


@pytest.mark.parametrize("title", [
    "Executive Coach",
    "Leadership Coach",
    "Career Coach",
    "Business Coach & Facilitator",
    "Trener Biznesu",
    "Coach Kariery",
    "L&D Specialist",
    "Learning and Development Manager",
    "Corporate Coach",
    "Agile Coach (Enterprise Leadership)",
    "Life Coach",
    "Senior Talent Development Coach",
    "Бізнес-коуч",
    "Коуч з лідерства та розвитку команд",
    # Corporate interim roles & corporate acronyms
    "Acting Executive Coach",
    "Acting Agile Coach",
    "Trener Biznesu - FinTech PSA",
])
def test_pre_filter_allows_valid_coaching_titles(title: str):
    res = pre_filter_job(title, VALID_DESC)
    assert res.passed is True
    assert res.reason is None


@pytest.mark.parametrize("empty_title", [
    "",
    "   ",
    "\n\t  \r",
    None,
])
def test_pre_filter_rejects_empty_titles(empty_title):
    res = pre_filter_job(empty_title, VALID_DESC)
    assert res.passed is False
    assert res.reason == "TITLE_EMPTY"


def test_pre_filter_rejects_stub_descriptions():
    # Under 150 characters
    res = pre_filter_job("Executive Coach", "Short stub description under 150 chars.")
    assert res.passed is False
    assert res.reason is not None
    assert "DESCRIPTION_TOO_SHORT" in res.reason
    assert "length=" in res.reason

    # Empty string
    res_empty = pre_filter_job("Executive Coach", "")
    assert res_empty.passed is False
    assert res_empty.reason is not None
    assert "DESCRIPTION_TOO_SHORT" in res_empty.reason

    # Whitespace only
    res_ws = pre_filter_job("Executive Coach", "    \n\t   ")
    assert res_ws.passed is False
    assert res_ws.reason is not None
    assert "DESCRIPTION_TOO_SHORT" in res_ws.reason


def test_pre_filter_rejects_descriptions_with_fewer_than_30_words():
    # Long words that exceed 150 characters total, but fewer than 30 words
    long_words = ["supercalifragilisticexpialidocious"] * 10
    desc_long_few_words = " ".join(long_words)
    assert len(desc_long_few_words) >= 150
    assert len(desc_long_few_words.split()) == 10

    res = pre_filter_job("Leadership Coach", desc_long_few_words)
    assert res.passed is False
    assert res.reason is not None
    assert "DESCRIPTION_TOO_SHORT" in res.reason
    assert "words=" in res.reason


def test_pre_filter_result_dataclass_immutability():
    res = PreFilterResult(passed=True, reason=None)
    assert res.passed is True
    assert res.reason is None
    with pytest.raises((AttributeError, TypeError)):
        res.passed = False  # type: ignore[misc]
