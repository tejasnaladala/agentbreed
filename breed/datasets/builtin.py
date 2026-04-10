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
    # === Technology (expansion -- resolved YES) ===
    (
        "builtin-tech-009",
        "Will OpenAI release a model with native voice conversation by end of 2024?",
        1.0, "tech", "2024-01-01", "2024-09-24",
    ),
    (
        "builtin-tech-010",
        "Will NVIDIA become the most valuable public company in the world at any point in 2024?",
        1.0, "tech", "2024-01-01", "2024-06-18",
    ),
    (
        "builtin-tech-011",
        "Will TSMC begin 3nm chip mass production by end of 2023?",
        1.0, "tech", "2022-01-01", "2023-01-01",
    ),
    (
        "builtin-tech-012",
        "Will GitHub Copilot reach 1 million paying subscribers by end of 2023?",
        1.0, "tech", "2023-01-01", "2023-10-31",
    ),
    (
        "builtin-tech-013",
        "Will Samsung release a flagship smartphone with on-device generative AI by end of 2024?",
        1.0, "tech", "2023-06-01", "2024-01-17",
    ),
    # === Technology (expansion -- resolved NO) ===
    (
        "builtin-tech-109",
        "Will a consumer AR headset from Meta or Apple ship under $500 by end of 2024?",
        0.0, "tech", "2023-06-01", "2024-12-31",
    ),
    (
        "builtin-tech-110",
        "Will any public LLM score above 50% on the ARC-AGI benchmark by end of 2024?",
        0.0, "tech", "2024-01-01", "2024-12-31",
    ),
    (
        "builtin-tech-111",
        "Will Boston Dynamics release a consumer home robot by end of 2024?",
        0.0, "tech", "2023-01-01", "2024-12-31",
    ),
    (
        "builtin-tech-112",
        "Will Google deprecate its third-party cookie support in Chrome by end of 2024?",
        0.0, "tech", "2023-01-01", "2024-12-31",
    ),
    # === Science (expansion -- resolved YES) ===
    (
        "builtin-sci-006",
        "Will NASA's Artemis I mission complete its lunar flyby by end of 2022?",
        1.0, "science", "2022-08-01", "2022-12-11",
    ),
    (
        "builtin-sci-007",
        "Will the World Health Organization declare mpox a public health emergency of international concern in 2022?",
        1.0, "science", "2022-06-01", "2022-07-23",
    ),
    (
        "builtin-sci-008",
        "Will OSIRIS-REx return its asteroid Bennu sample capsule to Earth in 2023?",
        1.0, "science", "2022-01-01", "2023-09-24",
    ),
    (
        "builtin-sci-009",
        "Will the Nobel Prize in Chemistry be awarded for work on protein structure prediction by end of 2024?",
        1.0, "science", "2024-01-01", "2024-10-09",
    ),
    (
        "builtin-sci-010",
        "Will India's Chandrayaan-3 successfully soft-land on the Moon by end of 2023?",
        1.0, "science", "2023-07-01", "2023-08-23",
    ),
    (
        "builtin-sci-011",
        "Will an AlphaFold successor release predictions for the full human proteome by end of 2022?",
        1.0, "science", "2021-12-01", "2022-07-28",
    ),
    (
        "builtin-sci-012",
        "Will a lecanemab-class Alzheimer's drug receive full FDA approval by end of 2023?",
        1.0, "science", "2022-06-01", "2023-07-06",
    ),
    # === Science (expansion -- resolved NO) ===
    (
        "builtin-sci-106",
        "Will SpaceX launch a crewed mission to Mars by end of 2024?",
        0.0, "science", "2022-01-01", "2024-12-31",
    ),
    (
        "builtin-sci-107",
        "Will a universal flu vaccine receive FDA approval by end of 2024?",
        0.0, "science", "2023-01-01", "2024-12-31",
    ),
    (
        "builtin-sci-108",
        "Will a cold fusion experiment be peer-reviewed and reproduced by end of 2024?",
        0.0, "science", "2023-01-01", "2024-12-31",
    ),
    # === Politics (expansion -- resolved YES) ===
    (
        "builtin-pol-006",
        "Will Finland officially join NATO by end of 2023?",
        1.0, "politics", "2022-05-01", "2023-04-04",
    ),
    (
        "builtin-pol-007",
        "Will Rishi Sunak become Prime Minister of the United Kingdom in 2022?",
        1.0, "politics", "2022-09-01", "2022-10-25",
    ),
    (
        "builtin-pol-008",
        "Will Sweden officially join NATO by end of 2024?",
        1.0, "politics", "2022-05-01", "2024-03-07",
    ),
    (
        "builtin-pol-009",
        "Will Lula da Silva be sworn in as president of Brazil in January 2023?",
        1.0, "politics", "2022-10-01", "2023-01-01",
    ),
    # === Politics (expansion -- resolved NO) ===
    (
        "builtin-pol-105",
        "Will a two-state Israel-Palestine peace agreement be signed by end of 2024?",
        0.0, "politics", "2023-10-01", "2024-12-31",
    ),
    (
        "builtin-pol-106",
        "Will the US Supreme Court overturn Section 230 protections by end of 2024?",
        0.0, "politics", "2023-01-01", "2024-12-31",
    ),
    # === Economics (expansion -- resolved YES) ===
    (
        "builtin-econ-006",
        "Will the Japanese Yen trade above 150 per USD at any point in 2024?",
        1.0, "economics", "2024-01-01", "2024-04-29",
    ),
    (
        "builtin-econ-007",
        "Will gold spot price exceed $2,500/oz in 2024?",
        1.0, "economics", "2024-01-01", "2024-08-16",
    ),
    (
        "builtin-econ-008",
        "Will the European Central Bank cut its key policy rate in 2024?",
        1.0, "economics", "2024-01-01", "2024-06-06",
    ),
    (
        "builtin-econ-009",
        "Will Saudi Aramco maintain its OPEC+ production cuts through mid-2024?",
        1.0, "economics", "2023-06-01", "2024-06-30",
    ),
    (
        "builtin-econ-010",
        "Will UK inflation (CPI) fall below the Bank of England's 2% target at any point in 2024?",
        1.0, "economics", "2023-09-01", "2024-05-22",
    ),
    (
        "builtin-econ-011",
        "Will Microsoft close its acquisition of Activision Blizzard by end of 2023?",
        1.0, "economics", "2022-01-01", "2023-10-13",
    ),
    (
        "builtin-econ-012",
        "Will Silicon Valley Bank be placed into FDIC receivership by end of 2023?",
        1.0, "economics", "2023-03-01", "2023-03-10",
    ),
    (
        "builtin-econ-013",
        "Will Credit Suisse cease to exist as an independent bank by end of 2023?",
        1.0, "economics", "2023-01-01", "2023-06-12",
    ),
    (
        "builtin-econ-014",
        "Will global Brent crude oil trade above $85/barrel at any point in 2024?",
        1.0, "economics", "2024-01-01", "2024-04-05",
    ),
    # === Economics (expansion -- resolved NO) ===
    (
        "builtin-econ-106",
        "Will US 30-year fixed mortgage rates fall below 5% in 2024?",
        0.0, "economics", "2024-01-01", "2024-12-31",
    ),
    (
        "builtin-econ-107",
        "Will the Chinese yuan be allowed to float freely against the US dollar by end of 2024?",
        0.0, "economics", "2023-01-01", "2024-12-31",
    ),
    (
        "builtin-econ-108",
        "Will China's GDP growth exceed 6% year-over-year in any quarter of 2024?",
        0.0, "economics", "2024-01-01", "2024-12-31",
    ),
    (
        "builtin-econ-109",
        "Will the Dow Jones Industrial Average close below 30,000 at any point in 2024?",
        0.0, "economics", "2024-01-01", "2024-12-31",
    ),
    (
        "builtin-econ-110",
        "Will the US federal funds rate reach 0% at any point in 2024?",
        0.0, "economics", "2024-01-01", "2024-12-31",
    ),
    (
        "builtin-econ-111",
        "Will the eurozone enter a technical recession in the first half of 2024?",
        0.0, "economics", "2024-01-01", "2024-07-31",
    ),
    # === Sports (expansion -- resolved YES) ===
    (
        "builtin-sports-004",
        "Will Spain win the UEFA Euro 2024 football tournament?",
        1.0, "sports", "2024-06-01", "2024-07-14",
    ),
    (
        "builtin-sports-005",
        "Will the Boston Celtics win the 2023-24 NBA championship?",
        1.0, "sports", "2023-10-01", "2024-06-17",
    ),
    (
        "builtin-sports-006",
        "Will Novak Djokovic win an Olympic gold medal in men's tennis singles at the 2024 Paris Olympics?",
        1.0, "sports", "2024-07-01", "2024-08-04",
    ),
    (
        "builtin-sports-007",
        "Will the Florida Panthers win the 2024 NHL Stanley Cup?",
        1.0, "sports", "2023-10-01", "2024-06-24",
    ),
    (
        "builtin-sports-008",
        "Will Max Verstappen win the 2023 Formula 1 Drivers' Championship?",
        1.0, "sports", "2023-03-01", "2023-10-07",
    ),
    (
        "builtin-sports-009",
        "Will Simone Biles win gold in the women's all-around gymnastics final at the 2024 Paris Olympics?",
        1.0, "sports", "2024-07-01", "2024-08-01",
    ),
    # === Sports (expansion -- resolved NO) ===
    (
        "builtin-sports-103",
        "Will France win the UEFA Euro 2024 football tournament?",
        0.0, "sports", "2024-06-01", "2024-07-14",
    ),
    (
        "builtin-sports-104",
        "Will Lewis Hamilton win the 2023 Formula 1 Drivers' Championship?",
        0.0, "sports", "2023-03-01", "2023-12-31",
    ),
    (
        "builtin-sports-105",
        "Will the San Francisco 49ers win Super Bowl LVIII (Feb 2024)?",
        0.0, "sports", "2024-01-01", "2024-02-11",
    ),
    (
        "builtin-sports-106",
        "Will Rafael Nadal win a Grand Slam singles title in 2024?",
        0.0, "sports", "2024-01-01", "2024-12-31",
    ),
    # === Culture (expansion -- resolved YES) ===
    (
        "builtin-culture-001",
        "Will 'Oppenheimer' win Best Picture at the 96th Academy Awards (2024)?",
        1.0, "culture", "2024-01-01", "2024-03-10",
    ),
    (
        "builtin-culture-002",
        "Will Taylor Swift's 'Eras Tour' become the highest-grossing concert tour of all time by end of 2023?",
        1.0, "culture", "2023-03-01", "2023-12-31",
    ),
    (
        "builtin-culture-003",
        "Will the film 'Barbie' gross over $1 billion worldwide by end of 2023?",
        1.0, "culture", "2023-07-01", "2023-08-06",
    ),
    (
        "builtin-culture-004",
        "Will the 2023 Hollywood writers' strike (WGA) conclude with a ratified deal by end of 2023?",
        1.0, "culture", "2023-05-01", "2023-10-09",
    ),
    (
        "builtin-culture-005",
        "Will 'Everything Everywhere All at Once' win Best Picture at the 95th Academy Awards (2023)?",
        1.0, "culture", "2023-01-01", "2023-03-12",
    ),
    (
        "builtin-culture-006",
        "Will Miley Cyrus's 'Flowers' reach #1 on the Billboard Hot 100 in 2023?",
        1.0, "culture", "2023-01-01", "2023-02-04",
    ),
    # === Culture (expansion -- resolved NO) ===
    (
        "builtin-culture-101",
        "Will 'Avatar: The Way of Water' gross more than 'Avengers: Endgame' worldwide by end of 2023?",
        0.0, "culture", "2023-01-01", "2023-12-31",
    ),
    (
        "builtin-culture-102",
        "Will a non-English-language film win Best Picture at the 96th Academy Awards (2024)?",
        0.0, "culture", "2024-01-01", "2024-03-10",
    ),
    (
        "builtin-culture-103",
        "Will Beyonce's 'Renaissance' win Album of the Year at the 2023 Grammy Awards?",
        0.0, "culture", "2023-01-01", "2023-02-05",
    ),
    (
        "builtin-culture-104",
        "Will 'Indiana Jones and the Dial of Destiny' gross over $500 million worldwide by end of 2023?",
        0.0, "culture", "2023-06-01", "2023-12-31",
    ),
    # === Finance (expansion -- resolved YES) ===
    (
        "builtin-finance-001",
        "Will the US SEC approve a spot Bitcoin ETF by end of January 2024?",
        1.0, "finance", "2023-06-01", "2024-01-10",
    ),
    (
        "builtin-finance-002",
        "Will the US SEC approve a spot Ethereum ETF by end of 2024?",
        1.0, "finance", "2024-01-01", "2024-05-23",
    ),
    (
        "builtin-finance-003",
        "Will FTX founder Sam Bankman-Fried be convicted on criminal fraud charges by end of 2023?",
        1.0, "finance", "2023-01-01", "2023-11-02",
    ),
    (
        "builtin-finance-004",
        "Will Moody's downgrade the US sovereign credit rating from Aaa by end of 2023?",
        1.0, "finance", "2023-01-01", "2023-11-10",
    ),
    (
        "builtin-finance-005",
        "Will Fitch downgrade the US sovereign credit rating from AAA in 2023?",
        1.0, "finance", "2023-01-01", "2023-08-01",
    ),
    (
        "builtin-finance-006",
        "Will Berkshire Hathaway's market capitalization exceed $1 trillion in 2024?",
        1.0, "finance", "2024-01-01", "2024-08-28",
    ),
    (
        "builtin-finance-007",
        "Will Eli Lilly's market capitalization exceed $700 billion at any point in 2024?",
        1.0, "finance", "2024-01-01", "2024-03-01",
    ),
    # === Finance (expansion -- resolved NO) ===
    (
        "builtin-finance-101",
        "Will any major central bank launch a retail CBDC for general public use by end of 2024?",
        0.0, "finance", "2023-01-01", "2024-12-31",
    ),
    (
        "builtin-finance-102",
        "Will the US national debt exceed $40 trillion by end of 2024?",
        0.0, "finance", "2024-01-01", "2024-12-31",
    ),
    (
        "builtin-finance-103",
        "Will Tether (USDT) lose its peg by more than 5% for 24+ hours in 2024?",
        0.0, "finance", "2024-01-01", "2024-12-31",
    ),
]


def load_builtin_questions() -> QuestionDataset:
    """Return the built-in resolved-questions dataset.

    This is a curated set of 120 historically resolved binary
    forecasting questions from 2022-2024, balanced across seven
    categories (tech, science, politics, economics, sports, culture,
    finance) with ~58% YES base rate. Intended for pipeline validation
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
        version="0.2.0",
        download_date="2026-04-09",
        n_questions=120,
        description=(
            "Curated set of 120 historically resolved binary forecasting "
            "questions across tech, science, politics, economics, sports, "
            "culture, and finance (2022-2024). Large enough for meaningful "
            "train/val/test splits. For pipeline validation and reproducible "
            "demos only -- not for final paper claims."
        ),
        citations=(),
    )
    return QuestionDataset(metadata=metadata, questions=questions)
