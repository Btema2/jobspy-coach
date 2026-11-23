"""Today's hard-coded pre-filter regexes and scoring-rubric wording, frozen as tracked data.

An imported (or migrated) profile is seeded from this module so behaviour is unchanged: the five regex
groups become ``pattern`` rules with their reason labels, and the rubric lines become the core-role
definition, adjacent-role definition and dealbreakers. This module deliberately imports nothing from the
pre-filter; the patterns are literal copies (a test keeps them equal to the live regexes).

``pattern`` rules are accepted by ``ops`` only when they equal an entry of ``SEED_RULES`` exactly.
The non-core score cap is not stored: it is ``wow - 0.1`` once a core-role definition exists.

The ``LEGACY_V1_*`` texts are the definitions as first imported (with a band header and fixed numbers); they
are kept only to recognise stored, unedited seed text. ``LEGACY_*`` (no ``V1``) are the current slot bodies the
rubric template wraps (``LEGACY_V2_*`` is the short-lived first Branch 3 wording); ``LEGACY_FIELD_UPGRADES`` maps each
profile field to its (earlier texts, current) pair.
"""
from dataclasses import dataclass
from typing import Dict, Tuple

from src.profile.snapshot import Rule


@dataclass(frozen=True)
class SeedRule:
    kind: str
    pattern: str
    reason: str


_EXCLUSION_TITLE = r'\b(sport[s]?|sportow[a-ząćęłńóśźż]*|спорт[а-яіїє]*|athletic[s]?|lekkoatletyk[ai]|swim|swimming|pływani[ae]|basen|плаванн[яі]|tennis|tenis[a-ząćęłńóśźż]*|padel|squash|теніс[а-яіїє]*|gym|fitness|siłowni[a-ząćęłńóśźż]*|kulturystyk[a-ząćęłńóśźż]*|workout|cross-?fit|фітнес[а-яіїє]*|soccer|football|piłk[ai]|piłkars[a-ząćęłńóśźż]*|futsal|футбол[а-яіїє]*|basketball|koszykówk[ai]|volleyball|siatkówk[ai]|баскетбол[а-яіїє]*|волейбол[а-яіїє]*|gymnastics|gimnastyk[ai]|acrobatics|akrobatyk[ai]|гімнастик[а-яіїє]*|hockey|hokej|skating|łyżw[a-ząćęłńóśźż]*|rolk[a-ząćęłńóśźż]*|хокей[а-яіїє]*|running|biegani[ae]|біг[а-яіїє]*|martial\s*arts|karate|judo|boxing|boks|mma|wrestling|zapasy|бокс[а-яіїє]*|карате|climbing|wspinaczk[ai]|ski|skiing|snowboard|narciar[a-ząćęłńóśźż]*|лиж[іи]|yoga|joga|pilates|йог[аи]|horse\s*riding|jeździectw[ao]|hipoterap[a-ząćęłńóśźż]*|кінн[а-яіїє]*|vocal|singing|śpiew[a-ząćęłńóśźż]*|głos|voice|вокал[а-яіїє]*|спів[а-яіїє]*|acting\s+coach|aktors[a-ząćęłńóśźż]*|drama|teatr|theatre|театр[а-яіїє]*|dance|taniec|tańca|choreograph[a-z]*|танц[іе][а-яіїє]*|e-?sport[s]?|gaming|valorant|league\s*of\s*legends|cs:go|dota|animal[s]?|pet[s]?|dog|puppy|pies|(?:treser|trener)[a-ząćęłńóśźż]*\s+ps[aó]w|treser[a-ząćęłńóśźż]*|zwierząt|собак[а-яіїє]*|цуцен[а-яіїє]*)\b'
_HARD_TRAP = r'\b(?:project\s+manager|product\s+manager|delivery\s+manager|scrum|agile|developer|devops|qa|account\s+executive|sales\s+manager|sales\s+director|financial\s+controller|audit\w*|fitness|gym|football|soccer|tennis|swim\w*|athletic\w*|vocal|bus|coach\s+builder|motor\s?coach|coach\s+driver|driver|programista|inżynier\s+oprogramowania|kierowca|розробник|водій)\b'
_SOFT_TRAP = r'\b(?:software|engineering|architects?|data|program(?:me)?\s+managers?)\b'
_SENIORITY = r'\b(?:directors?|vp|vice\s+presidents?|head\s+of|chief|ceo|cfo|coo|cto|cio|senior\s+managers?|dyrektor\w*|kierownik\w*|prezes\w*|członek\s+zarządu|директор\w*|керівник\w*|голова)\b'
_RESCUE = r'(?<!\w)(?:coach\w*|psycholog\w*|counsel\w*|doradc\w*|trener\w*|mentor\w*|learning\s+(?:and|&)\s+development|l&d|talent\s+development|people\s+development|learning\s+experience|leadership\s+development|organi[sz]ational\s+development|szkole\w*|психолог\w*|коуч\w*)(?!\w)'

SEED_RULES: Tuple[SeedRule, ...] = (
    SeedRule("block", _EXCLUSION_TITLE, "EXCLUDED_TITLE_REGEX"),
    SeedRule("block", _HARD_TRAP, "EXCLUDED_TITLE_TRAP"),
    SeedRule("soft", _SOFT_TRAP, "EXCLUDED_TITLE_TRAP"),
    SeedRule("seniority", _SENIORITY, "EXCLUDED_SENIORITY"),
    SeedRule("rescue", _RESCUE, "RESCUE"),
)


def seed_rules() -> Tuple[Rule, ...]:
    """The seed as profile rules (``pattern`` mode, source ``seed``)."""
    return tuple(Rule(s.kind, s.pattern, "pattern", s.reason, "seed") for s in SEED_RULES)


LEGACY_V1_CORE_ROLE_DEFINITION = """- WOW (8.5 - 10.0): Only direct 1-on-1 (or small-group) coaching or psychological-support roles, such as a coach, associate coach on a coaching platform, career coach or advisor, wellbeing coach, corporate psychologist, or a career / livelihoods coach for an NGO or education organisation, with compatible location and language terms. Set `role_is_direct_coaching` to true for these roles. WOW also requires `role_is_direct_coaching` to be true: a role that is not direct coaching never scores 8.5 or above.
   - `role_is_direct_coaching` is true only when the role's main duty is delivering 1-on-1 (or small-group) coaching or psychological support to clients. It is false for a corporate trainer, a learning and development (L&D) specialist, leadership-development or programme / people-development roles, an internal training facilitator, and any role whose main duty is designing, running or coordinating training programmes."""

LEGACY_V1_ADJACENT_ROLE_DEFINITION = """- DIGEST (6.5 - 8.4): Corporate learning and development specialist, talent development, people development, training coordinator, or career counselling / advisory roles where the candidate's background is transferable and the required language level is compatible with the candidate's languages."""

LEGACY_V1_DEALBREAKERS = """- HARD REJECT (1.0 - 2.0): Coaching or people development is only a secondary duty of an IT / Engineering / Sales / Project Management / Scrum Master role; the role requires coding, technical IT delivery or P&L management; the role is executive coaching of C-suite leaders that strictly requires native / C2-level proficiency in a language the candidate does not have at that level; or the required seniority is beyond what the candidate profile supports. Set `hard_reject` to true."""


# V2: the first Branch 3 wording. It still held commands ("Score as MARGINAL instead", "It is not a core role
# for ...") inside the data block, so the data notice contradicted it; kept only to recognise stored text.
LEGACY_V2_CORE_ROLE_DEFINITION = """Only direct 1-on-1 (or small-group) coaching or psychological-support roles, such as a coach, associate coach on a coaching platform, career coach or advisor, wellbeing coach, corporate psychologist, or a career / livelihoods coach for an NGO or education organisation.
A role is a core role only when the role's main duty is delivering 1-on-1 (or small-group) coaching or psychological support to clients. It is not a core role for a corporate trainer, a learning and development (L&D) specialist, leadership-development or programme / people-development roles, an internal training facilitator, and any role whose main duty is designing, running or coordinating training programmes."""

LEGACY_V2_ADJACENT_ROLE_DEFINITION = """Corporate learning and development specialist, talent development, people development, training coordinator, or career counselling / advisory roles where the candidate's background is transferable and the required language level is compatible with the candidate's languages. Score as MARGINAL instead: mixed HR / recruitment responsibilities, unclear coaching scope, administrative L&D."""

# Current bodies: declarative definitions (what a role is), never commands to the model.
LEGACY_CORE_ROLE_DEFINITION = """Only direct 1-on-1 (or small-group) coaching or psychological-support roles, such as a coach, associate coach on a coaching platform, career coach or advisor, wellbeing coach, corporate psychologist, or a career / livelihoods coach for an NGO or education organisation.
A role is a core role when its main duty is delivering 1-on-1 (or small-group) coaching or psychological support to clients. A corporate trainer, a learning and development (L&D) specialist, leadership-development and programme / people-development roles, an internal training facilitator, and any role whose main duty is designing, running or coordinating training programmes are not core roles."""

LEGACY_ADJACENT_ROLE_DEFINITION = """Corporate learning and development specialist, talent development, people development, training coordinator, or career counselling / advisory roles where the candidate's background is transferable and the required language level is compatible with the candidate's languages.
Roles with mixed HR / recruitment responsibilities, an unclear coaching scope or an administrative L&D focus are marginal matches, not adjacent roles."""

LEGACY_DEALBREAKERS = """Coaching or people development is only a secondary duty of an IT / Engineering / Sales / Project Management / Scrum Master role; the role requires coding, technical IT delivery or P&L management; the role is executive coaching of C-suite leaders that strictly requires native / C2-level proficiency in a language the candidate does not have at that level; or the required seniority is beyond what the candidate profile supports."""

LEGACY_FIELD_UPGRADES: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "core_role_definition": ((LEGACY_V1_CORE_ROLE_DEFINITION, LEGACY_V2_CORE_ROLE_DEFINITION), LEGACY_CORE_ROLE_DEFINITION),
    "adjacent_role_definition": (
        (LEGACY_V1_ADJACENT_ROLE_DEFINITION, LEGACY_V2_ADJACENT_ROLE_DEFINITION),
        LEGACY_ADJACENT_ROLE_DEFINITION,
    ),
    "dealbreakers": ((LEGACY_V1_DEALBREAKERS,), LEGACY_DEALBREAKERS),
}

# Earlier seed text -> current seed body (an exact match means the text was never edited).
LEGACY_TEXT_UPGRADES: Dict[str, str] = {
    old: new for olds, new in LEGACY_FIELD_UPGRADES.values() for old in olds
}
