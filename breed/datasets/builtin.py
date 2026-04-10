"""Built-in synthetic forecasting dataset for offline experimentation.

This dataset is designed for pipeline validation and reproducible
demos. It is NOT suitable for the final paper results -- those require
real Metaculus or ForecastBench data. The questions here have known
historical outcomes and are scoped to events resolved by 2025-12-31.
"""

from __future__ import annotations

from breed.datasets.base import (
    DatasetMetadata,
    ForecastingQuestion,
    QuestionDataset,
)

# ---------------------------------------------------------------------------
# Question pool -- resolved binary questions, manually curated
# ---------------------------------------------------------------------------
# Format: (question_id, prompt, outcome, category, asked_date, resolved_date)

_BUILTIN_POOL: list[tuple[str, str, float, str, str, str]] = [
    # === Technology (resolved YES) ===
    (
        "builtin-tech-001",
        "Will SpaceX Starship complete an orbital flight by end of 2024?",
        1.0, "tech", "2023-04-01", "2024-10-13",
    ),
    (
        "builtin-tech-002",
        "Will Apple release a mixed-reality headset by end of 2024?",
        1.0, "tech", "2022-06-01", "2024-02-02",
    ),
    (
        "builtin-tech-003",
        "Will an LLM exceed 90% on MMLU by end of 2024?",
        1.0, "tech", "2023-01-01", "2024-04-09",
    ),
    (
        "builtin-tech-004",
        "Will at least one AI lab release a model with native vision by end of 2024?",
        1.0, "tech", "2023-03-01", "2023-09-25",
    ),
    (
        "builtin-tech-005",
        "Will a Level-4 robotaxi service operate in a major US city by end of 2024?",
        1.0, "tech", "2022-01-01", "2024-08-01",
    ),
    (
        "builtin-tech-006",
        "Will Anthropic release a model called Claude 3 by end of 2024?",
        1.0, "tech", "2023-05-01", "2024-03-04",
    ),
    (
        "builtin-tech-007",
        "Will Meta release an open-weights LLM with at least 70B parameters by end of 2024?",
        1.0, "tech", "2023-02-01", "2024-04-18",
    ),
    (
        "builtin-tech-008",
        "Will ChatGPT exceed 100 million weekly active users by end of 2023?",
        1.0, "tech", "2023-01-15", "2023-11-01",
    ),
    # === Technology (resolved NO) ===
    (
        "builtin-tech-101",
        "Will OpenAI release GPT-5 by end of 2024?",
        0.0, "tech", "2023-03-01", "2024-12-31",
    ),
    (
        "builtin-tech-102",
        "Will quantum computers break RSA-2048 encryption by end of 2025?",
        0.0, "tech", "2022-01-01", "2025-12-31",
    ),
    (
        "builtin-tech-103",
        "Will a humanoid robot be commercially available for under $20,000 by end of 2024?",
        0.0, "tech", "2023-01-01", "2024-12-31",
    ),
    (
        "builtin-tech-104",
        "Will Tesla deliver Cybertruck to more than 100,000 customers by end of 2024?",
        0.0, "tech", "2023-06-01", "2024-12-31",
    ),
    (
        "builtin-tech-105",
        "Will a fully self-driving Tesla without driver supervision be approved by US regulators by end of 2024?",
        0.0, "tech", "2023-01-01", "2024-12-31",
    ),
    (
        "builtin-tech-106",
        "Will Google's Gemini Ultra outperform GPT-4 on MMLU by end of 2023?",
        0.0, "tech", "2023-09-01", "2023-12-31",
    ),
    (
        "builtin-tech-107",
        "Will Apple release an AI-powered Siri rebuild by end of 2024?",
        0.0, "tech", "2024-01-01", "2024-12-31",
    ),
    (
        "builtin-tech-108",
        "Will ChatGPT-5 (named GPT-5) be released by end of 2024?",
        0.0, "tech", "2023-06-01", "2024-12-31",
    ),
    # === Science (resolved YES) ===
    (
        "builtin-sci-001",
        "Will nuclear fusion achieve net energy gain in a lab experiment by end of 2023?",
        1.0, "science", "2022-01-01", "2022-12-13",
    ),
    (
        "builtin-sci-002",
        "Will CRISPR gene therapy receive FDA approval for sickle cell disease by end of 2024?",
        1.0, "science", "2022-06-01", "2023-12-08",
    ),
    (
        "builtin-sci-003",
        "Will the JWST observe a galaxy at redshift z>14 by end of 2024?",
        1.0, "science", "2022-07-01", "2024-05-30",
    ),
    (
        "builtin-sci-004",
        "Will an mRNA cancer vaccine reach Phase 3 trials by end of 2024?",
        1.0, "science", "2023-01-01", "2024-04-01",
    ),
    (
        "builtin-sci-005",
        "Will a room-temperature superconductor claim be debunked within 6 months of announcement?",
        1.0, "science", "2023-07-01", "2023-12-01",
    ),
    # === Science (resolved NO) ===
    (
        "builtin-sci-101",
        "Will the first pig-to-human heart transplant recipient survive more than 2 years?",
        0.0, "science", "2022-01-01", "2024-01-15",
    ),
    (
        "builtin-sci-102",
        "Will JWST discover unambiguous biosignatures on an exoplanet by end of 2024?",
        0.0, "science", "2022-07-01", "2024-12-31",
    ),
    (
        "builtin-sci-103",
        "Will a private asteroid mining mission launch by end of 2024?",
        0.0, "science", "2022-01-01", "2024-12-31",
    ),
    (
        "builtin-sci-104",
        "Will SETI confirm a non-natural radio signal by end of 2024?",
        0.0, "science", "2022-01-01", "2024-12-31",
    ),
    (
        "builtin-sci-105",
        "Will room-temperature superconductivity be reproduced and confirmed by end of 2024?",
        0.0, "science", "2023-08-01", "2024-12-31",
    ),
    # === Politics (resolved YES) ===
    (
        "builtin-pol-001",
        "Will Donald Trump win the 2024 US presidential election?",
        1.0, "politics", "2024-01-01", "2024-11-06",
    ),
    (
        "builtin-pol-002",
        "Will the UK Labour Party form a government after the next general election (held by 2025)?",
        1.0, "politics", "2024-01-01", "2024-07-05",
    ),
    (
        "builtin-pol-003",
        "Will Argentina elect Javier Milei as president in 2023?",
        1.0, "politics", "2023-08-01", "2023-11-19",
    ),
    (
        "builtin-pol-004",
        "Will the US debt ceiling be raised before June 2023?",
        1.0, "politics", "2023-04-01", "2023-06-03",
    ),
    (
        "builtin-pol-005",
        "Will the EU pass the AI Act into law by end of 2024?",
        1.0, "politics", "2023-06-01", "2024-08-01",
    ),
    # === Politics (resolved NO) ===
    (
        "builtin-pol-101",
        "Will Joe Biden win the 2024 US presidential election?",
        0.0, "politics", "2024-01-01", "2024-11-06",
    ),
    (
        "builtin-pol-102",
        "Will the US Congress pass comprehensive AI regulation legislation by end of 2024?",
        0.0, "politics", "2023-06-01", "2024-12-31",
    ),
    (
        "builtin-pol-103",
        "Will Russia and Ukraine sign a ceasefire agreement by end of 2024?",
        0.0, "politics", "2023-01-01", "2024-12-31",
    ),
    (
        "builtin-pol-104",
        "Will TikTok be banned in the US by end of 2024?",
        0.0, "politics", "2023-03-01", "2024-12-31",
    ),
    # === Economics (resolved YES) ===
    (
        "builtin-econ-001",
        "Will the US Federal Reserve cut interest rates in 2024?",
        1.0, "economics", "2024-01-01", "2024-09-18",
    ),
    (
        "builtin-econ-002",
        "Will Bitcoin exceed $50,000 in 2024?",
        1.0, "economics", "2023-12-01", "2024-02-12",
    ),
    (
        "builtin-econ-003",
        "Will the S&P 500 close above 5,000 by end of 2024?",
        1.0, "economics", "2024-01-01", "2024-02-09",
    ),
    (
        "builtin-econ-004",
        "Will Nvidia exceed a $2 trillion market cap by end of 2024?",
        1.0, "economics", "2024-01-01", "2024-02-23",
    ),
    (
        "builtin-econ-005",
        "Will US inflation (CPI) be below 4% year-over-year by end of 2024?",
        1.0, "economics", "2023-09-01", "2024-12-11",
    ),
    # === Economics (resolved NO) ===
    (
        "builtin-econ-101",
        "Will Bitcoin exceed $200,000 by end of 2024?",
        0.0, "economics", "2024-01-01", "2024-12-31",
    ),
    (
        "builtin-econ-102",
        "Will the US enter a recession (NBER definition) in 2023?",
        0.0, "economics", "2023-01-01", "2023-12-31",
    ),
    (
        "builtin-econ-103",
        "Will US unemployment exceed 5% in 2024?",
        0.0, "economics", "2024-01-01", "2024-12-31",
    ),
    (
        "builtin-econ-104",
        "Will Tesla stock close above $500 in 2024?",
        0.0, "economics", "2024-01-01", "2024-12-31",
    ),
    (
        "builtin-econ-105",
        "Will Apple's market cap fall below $2 trillion in 2024?",
        0.0, "economics", "2024-01-01", "2024-12-31",
    ),
    # === Sports / culture (resolved YES) ===
    (
        "builtin-sports-001",
        "Will Kansas City Chiefs win Super Bowl LVIII (Feb 2024)?",
        1.0, "sports", "2024-01-01", "2024-02-11",
    ),
    (
        "builtin-sports-002",
        "Will Real Madrid win the 2023-24 UEFA Champions League?",
        1.0, "sports", "2023-09-01", "2024-06-01",
    ),
    (
        "builtin-sports-003",
        "Will Argentina win the 2022 FIFA World Cup?",
        1.0, "sports", "2022-11-20", "2022-12-18",
    ),
    # === Sports (resolved NO) ===
    (
        "builtin-sports-101",
        "Will the LA Lakers win the 2023-24 NBA championship?",
        0.0, "sports", "2023-10-01", "2024-06-17",
    ),
    (
        "builtin-sports-102",
        "Will Manchester City win the 2023-24 UEFA Champions League?",
        0.0, "sports", "2023-09-01", "2024-06-01",
    ),
]


def load_builtin_questions() -> QuestionDataset:
    """Return the built-in resolved-questions dataset.

    This is a curated set of ~50 binary questions resolved between
    2022 and 2024, balanced across categories. Use for pipeline tests
    and reproducible demos -- NOT for final paper results.
    """
    questions = tuple(
        ForecastingQuestion(
            question_id=qid,
            prompt=prompt,
            outcome=outcome,
            category=category,
            asked_date=asked,
            resolved_date=resolved,
            source="builtin",
            metadata={},
        )
        for qid, prompt, outcome, category, asked, resolved in _BUILTIN_POOL
    )
    metadata = DatasetMetadata(
        name="agentbreed-builtin",
        source="agentbreed-research-package",
        url="https://github.com/agentbreed/agentbreed",
        license="MIT",
        version="0.1.0",
        download_date="2026-04-09",
        n_questions=len(questions),
        description=(
            "Curated set of historically resolved binary forecasting questions "
            "across tech, science, politics, economics, and sports. For pipeline "
            "validation and reproducible demos only -- not for final paper claims."
        ),
        citations=(),
    )
    return QuestionDataset(metadata=metadata, questions=questions)
