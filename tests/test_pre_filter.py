import pytest
from src.filters import classify_title
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
    "Life Coach",
    "Senior Talent Development Coach",
    "Бізнес-коуч",
    "Коуч з лідерства та розвитку команд",
    # Corporate interim roles & corporate acronyms
    "Acting Executive Coach",
    "Trener Biznesu - FinTech PSA",
    # Must-pass titles guarding against substring / over-broad trap matches
    "Transport Coordinator",
    "Business Coach",
    "Career Coach, Data Science Bootcamp",
    "Career Coach – Software Engineering Bootcamp",
    "Learning Experience Architect",
    "Learning and Development Specialist",
    "Doradca zawodowy",
    "Psycholog",
    "Psychologist",
    "Head of Coaching",
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


# ---------------------------------------------------------------------------
# Title traps and seniority (classify_title)
# ---------------------------------------------------------------------------

HARD_TRAP_TITLES = [
    "Project Manager",
    "Senior Project Manager",
    "Product Manager",
    "Scrum Master",
    "Agile Coach",
    "Agile Coach (Enterprise Leadership)",
    "Acting Agile Coach",
    "Delivery Manager",
    "Java Developer",
    "DevOps Engineer",
    "QA Analyst",
    "Account Executive",
    "Sales Manager",
    "Sales Director",
    "Financial Controller",
    "Internal Audit Specialist",
    "Bus Driver",
    "Coach Builder",
    "Motor Coach Operator",
    "Motorcoach Operator",
    "Coach Driver",
    "Delivery Driver",
    # PL / UK equivalents
    "Programista Python",
    "Inżynier oprogramowania",
    "Kierowca kategorii C",
    "Розробник програмного забезпечення",
    "Водій-експедитор",
]

SOFT_TRAP_TITLES = [
    "Software Engineer",
    "Software Architect",
    "Engineering Manager",
    "Solutions Architect",
    "Data Analyst",
    "Data Engineer",
    "Program Manager",
]

SOFT_TRAP_RESCUED_TITLES = [
    "Data Coach",
    "Software Engineering Mentor",
    "Career Coach – Software Engineering Bootcamp",
    "Career Coach, Data Science Bootcamp",
    "Program Manager, Learning and Development",
    "Program Manager, Learning & Development",
    "L&D Program Manager",
    "Talent Development Architect",
    "People Development Architect",
    "Learning Experience Architect",
    "Data Psychologist",
    "Engineering Counselor",
    "Architect Coaching Programme",
    "Trener Software Engineering",
    "Doradca ds. danych (data)",
    "Психолог data-команди",
    "Коуч для software-команд",
]

SENIORITY_TITLES = [
    "Senior Director, Paid Media",
    "Director of Machine Learning",
    "Director of Operations",
    "VP of People",
    "Vice President, Talent",
    "Head of Operations",
    "Chief People Officer",
    "CEO",
    "CFO Office Lead",
    "COO",
    "CTO",
    "CIO",
    "Senior Manager, Operations",
    # PL / UK
    "Dyrektor Sprzedaży",
    "Kierownik Projektu",
    "Prezes Zarządu",
    "Członek zarządu ds. finansów",
    "Директор з розвитку",
    "Керівник відділу продажів",
    "Голова правління",
]

SENIORITY_RESCUED_TITLES = [
    "Head of Coaching",
    "Director of Coaching",
    "Director, Learning and Development",
    "Director of Learning & Development",
    "VP, Talent Development",
    "Chief Psychologist",
    "Senior Manager, People Development",
    "Head of Mentoring",
    "Director of Counseling Services",
    "Dyrektor ds. rozwoju trenerów",
    "Kierownik zespołu psychologów",
    "Prezes fundacji doradców zawodowych",
    "Директор центру психології",
    "Керівник відділу коучингу",
]


@pytest.mark.parametrize("title", HARD_TRAP_TITLES)
def test_hard_traps_drop(title: str):
    reason = classify_title(title)
    assert reason is not None
    assert reason.startswith("EXCLUDED_TITLE_TRAP: '")
    res = pre_filter_job(title, VALID_DESC)
    assert res.passed is False
    assert res.reason == reason


@pytest.mark.parametrize("title", [
    "Agile Coach",
    "Scrum Master & Coach",
    "Project Manager, Learning and Development",
    "Product Manager / Mentor",
    "Delivery Manager Coach",
    "Developer Coach",
    "Sales Manager Psycholog",
    "Driver Trainer Counselor",
])
def test_hard_traps_are_never_rescued(title: str):
    assert classify_title(title) is not None


@pytest.mark.parametrize("title", SOFT_TRAP_TITLES)
def test_soft_traps_drop_without_rescue(title: str):
    reason = classify_title(title)
    assert reason is not None
    assert reason.startswith("EXCLUDED_TITLE_TRAP: '")
    assert pre_filter_job(title, VALID_DESC).passed is False


@pytest.mark.parametrize("title", SOFT_TRAP_RESCUED_TITLES)
def test_soft_traps_are_rescued_by_coaching_words(title: str):
    assert classify_title(title) is None
    assert pre_filter_job(title, VALID_DESC).passed is True


@pytest.mark.parametrize("title", SENIORITY_TITLES)
def test_seniority_drops(title: str):
    reason = classify_title(title)
    assert reason is not None
    assert reason.startswith("EXCLUDED_SENIORITY: '")
    res = pre_filter_job(title, VALID_DESC)
    assert res.passed is False
    assert res.reason == reason


@pytest.mark.parametrize("title", SENIORITY_RESCUED_TITLES)
def test_seniority_is_rescued_by_coaching_words(title: str):
    assert classify_title(title) is None
    assert pre_filter_job(title, VALID_DESC).passed is True


def test_bare_learning_does_not_rescue():
    assert classify_title("Director of Machine Learning") is not None
    assert classify_title("Head of Learning Systems") is not None


@pytest.mark.parametrize("title", [
    "Transport Coordinator",
    "Business Coach",
    "Business Analyst Coach",
    "Career Coach",
    "Acting Executive Coach",
    "Senior Talent Development Coach",
    "Metadata Analyst",        # 'data' only as a whole word
    "Directorate Liaison",  # 'director' only as a whole word
    "Architectural Historian",
    "Chiefdom Researcher",
    "Subusiness Mentor",
])
def test_whole_word_matching_never_uses_substrings(title: str):
    assert classify_title(title) is None


@pytest.mark.parametrize("title", ["DIRECTOR OF SALES", "pRoJeCt MaNaGeR", "SOFTWARE ENGINEER"])
def test_matching_is_case_insensitive(title: str):
    assert classify_title(title) is not None


@pytest.mark.parametrize("title", [None, "", "   "])
def test_classify_title_handles_empty(title):
    assert classify_title(title) is None


def test_classify_title_keeps_sports_reason_string():
    assert classify_title("Swimming Coach") == "EXCLUDED_TITLE_REGEX: 'Swimming'"


def test_traps_run_before_description_length_check():
    res = pre_filter_job("Project Manager", "short")
    assert res.reason is not None
    assert res.reason.startswith("EXCLUDED_TITLE_TRAP")


def test_sports_regex_runs_before_traps():
    res = pre_filter_job("Football Developer", VALID_DESC)
    assert res.reason is not None
    assert res.reason.startswith("EXCLUDED_TITLE_REGEX")


def test_empty_title_runs_first():
    assert pre_filter_job("", "short").reason == "TITLE_EMPTY"


def test_classify_title_is_exported_from_package():
    import src.filters as filters

    assert "classify_title" in filters.__all__
    assert filters.classify_title is classify_title


@pytest.mark.parametrize("title", [
    "Career Coach (Developers)",
    "Leadership Coach for Developers",
    "Coach for Software Developers",
    "Leadership Coach – Product Managers",
    "Mentor dla programistów",
    "Coach for Project Managers",
    "Career Coach for Delivery Managers",
    "Coaching for Sales Managers",
    "Career Coach for Drivers",
])
def test_plural_audience_words_do_not_trigger_hard_traps(title: str):
    assert classify_title(title) is None
    assert pre_filter_job(title, VALID_DESC).passed is True


@pytest.mark.parametrize("title", [
    "Developer",
    "Product Manager",
    "Project Manager",
    "Delivery Manager",
    "Sales Manager",
    "Programista",
    "Kierowca",
    "Driver",
])
def test_singular_hard_traps_still_drop(title: str):
    reason = classify_title(title)
    assert reason is not None
    assert reason.startswith("EXCLUDED_TITLE_TRAP")


@pytest.mark.parametrize("title", [
    "Head of Leadership Development",
    "Director, Leadership Development",
    "Head of Organizational Development",
    "Head of Organisational Development",
    "Leadership Development Program Manager",
    "Programme Manager, Leadership Development",
    "Kierownik ds. szkoleń",
    "Dyrektor ds. szkoleń i rozwoju",
])
def test_core_ld_phrasings_rescue(title: str):
    assert classify_title(title) is None
    assert pre_filter_job(title, VALID_DESC).passed is True


def test_ld_rescue_does_not_rescue_hard_traps_or_bare_learning():
    assert classify_title("Director of Machine Learning") is not None
    assert classify_title("Project Manager, Leadership Development") is not None
