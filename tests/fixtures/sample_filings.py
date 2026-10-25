"""
Fixtures generator for 10-K filings across 3 tickers (AAPL, NVDA, MSFT).
Used for deterministic offline testing of Item 1A parser and EDGAR data layer.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

SAMPLE_FILINGS = [
    # 1. AAPL 2023 10-K (HTML style)
    {
        "ticker": "AAPL",
        "form": "10-K",
        "accession": "0000320193-23-000106",
        "accepted_at_str": "2023-11-03 18:04:45",
        "raw_text": """
        <html><body>
        <div>PART I</div>
        <div><b>ITEM 1. BUSINESS</b></div>
        <p>Apple designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories.</p>
        <div><b>ITEM 1A. RISK FACTORS</b></div>
        <p>The company's business, reputation, results of operations and financial condition could be materially adversely affected by risks related to global economic conditions. We face intense competition in global markets characterized by aggressive price competition and rapid technological advances. Furthermore, supply chain disruptions could impair our ability to manufacture products.</p>
        <div><b>ITEM 1B. UNRESOLVED STAFF COMMENTS</b></div>
        <p>None.</p>
        <div><b>ITEM 2. PROPERTIES</b></div>
        <p>The company's headquarters are located in Cupertino, California.</p>
        </body></html>
        """,
        "expected_risk_keyword": "intense competition in global markets",
    },
    # 2. AAPL 2022 10-K (Text style with dashes)
    {
        "ticker": "AAPL",
        "form": "10-K",
        "accession": "0000320193-22-000108",
        "accepted_at_str": "2022-10-28 18:01:14",
        "raw_text": """
        ITEM 1. BUSINESS
        General overview of consumer electronics operations.

        ITEM 1A - RISK FACTORS
        Our operations are subject to risks of geopolitical tensions and international trade restrictions. New regulatory requirements in the European Union regarding third-party app stores may harm our Services business margins.

        ITEM 1B - UNRESOLVED STAFF COMMENTS
        There are no unresolved comments.
        """,
        "expected_risk_keyword": "geopolitical tensions and international trade restrictions",
    },
    # 3. AAPL 2021 10-K (Numbered heading style)
    {
        "ticker": "AAPL",
        "form": "10-K",
        "accession": "0000320193-21-000105",
        "accepted_at_str": "2021-10-29 18:02:10",
        "raw_text": """
        PART I
        Item 1. Business.
        Apple designs smartphones and computers.

        Item 1A. Risk Factors.
        Global semiconductor component shortages and freight capacity constraints could materially affect our product availability during peak holiday quarters. Component pricing volatility also affects gross margin.

        Item 1B. Unresolved Staff Comments.
        None.
        """,
        "expected_risk_keyword": "semiconductor component shortages",
    },
    # 4. NVDA 2024 10-K (Modern HTML SEC format)
    {
        "ticker": "NVDA",
        "form": "10-K",
        "accession": "0001045810-24-000029",
        "accepted_at_str": "2024-02-21 16:42:09",
        "raw_text": """
        <html><body>
        <h2>Item 1. Business</h2>
        <p>NVIDIA pioneered accelerated computing to tackle challenges ordinary computers cannot.</p>
        <h2>Item 1A. Risk Factors</h2>
        <p>Failure to meet the rapid surges in demand for our GPU architectures and AI networking fabrics could adversely impact customer relationships. We rely heavily on TSMC for our advanced wafer fabrication and packaging.</p>
        <h2>Item 1B. Unresolved Staff Comments</h2>
        <p>None.</p>
        </body></html>
        """,
        "expected_risk_keyword": "GPU architectures and AI networking fabrics",
    },
    # 5. NVDA 2023 10-K (Uppercase text format)
    {
        "ticker": "NVDA",
        "form": "10-K",
        "accession": "0001045810-23-000017",
        "accepted_at_str": "2023-03-03 16:30:00",
        "raw_text": """
        ITEM 1. BUSINESS
        NVIDIA designs graphics processing units.

        ITEM 1A. RISK FACTORS
        Export control restrictions imposed by the U.S. government on shipments of high-performance computing chips to China may significantly impact our future data center revenue growth and market share.

        ITEM 2. PROPERTIES
        Our corporate campus is in Santa Clara, California.
        """,
        "expected_risk_keyword": "Export control restrictions imposed by the U.S. government",
    },
    # 6. NVDA 2022 10-K (Mixed case format)
    {
        "ticker": "NVDA",
        "form": "10-K",
        "accession": "0001045810-22-000036",
        "accepted_at_str": "2022-03-18 16:35:12",
        "raw_text": """
        Item 1. Business
        Visual computing and accelerated platforms.

        Item 1A. Risk Factors
        The termination of our proposed acquisition of Arm Limited resulted in an operating charge of $1.36 billion. Volatility in cryptocurrency mining demand has historically caused volatility in gaming GPU sales.

        Item 1B. Unresolved Staff Comments
        None.
        """,
        "expected_risk_keyword": "acquisition of Arm Limited",
    },
    # 7. MSFT 2024 10-K (HTML with nested spans)
    {
        "ticker": "MSFT",
        "form": "10-K",
        "accession": "0000950170-24-087843",
        "accepted_at_str": "2024-07-30 16:15:33",
        "raw_text": """
        <div><span>PART I</span></div>
        <div><span>Item 1. Business</span></div>
        <p>Microsoft enables digital transformation for the era of an intelligent cloud and intelligent edge.</p>
        <div><span>Item 1A. Risk Factors</span></div>
        <p>Our significant capital expenditures in AI infrastructure and data center capacity may not yield anticipated commercial returns. Additionally, third-party cyberattacks targeting Azure cloud infrastructure pose catastrophic operational risks.</p>
        <div><span>Item 1B. Unresolved Staff Comments</span></div>
        <p>None.</p>
        """,
        "expected_risk_keyword": "capital expenditures in AI infrastructure",
    },
    # 8. MSFT 2023 10-K (Colon format)
    {
        "ticker": "MSFT",
        "form": "10-K",
        "accession": "0000950170-23-035122",
        "accepted_at_str": "2023-07-27 16:10:02",
        "raw_text": """
        Item 1: Business
        Cloud computing and software platforms.

        Item 1A: Risk Factors
        We face intense competition across all of our business segments, including cloud hyperscalers like AWS and Google Cloud. Complex regulatory reviews of our Activision Blizzard transaction could result in concessions or operational constraints.

        Item 1B: Unresolved Staff Comments
        None.
        """,
        "expected_risk_keyword": "Activision Blizzard transaction",
    },
    # 9. MSFT 2022 10-K (Standard formal filing)
    {
        "ticker": "MSFT",
        "form": "10-K",
        "accession": "0000950170-22-012817",
        "accepted_at_str": "2022-07-28 16:12:45",
        "raw_text": """
        ITEM 1. BUSINESS.
        Microsoft Corporation overview.

        ITEM 1A. RISK FACTORS.
        Adverse economic conditions, inflation, foreign currency exchange rate fluctuations, and supply chain constraints may impact enterprise IT software spend and hardware sales.

        ITEM 1B. UNRESOLVED STAFF COMMENTS.
        None.
        """,
        "expected_risk_keyword": "foreign currency exchange rate fluctuations",
    },
    # 10. AAPL 2020 10-K (Pandemic risk disclosure)
    {
        "ticker": "AAPL",
        "form": "10-K",
        "accession": "0000320193-20-000096",
        "accepted_at_str": "2020-10-30 18:00:22",
        "raw_text": """
        ITEM 1. BUSINESS
        Consumer technology products and digital services.

        ITEM 1A. RISK FACTORS
        The COVID-19 pandemic has disrupted our global retail store operations, supply chains, and consumer demand in retail locations. Temporary store closures and employee travel restrictions have impacted product launch timelines.

        ITEM 1B. UNRESOLVED STAFF COMMENTS
        None.
        """,
        "expected_risk_keyword": "COVID-19 pandemic has disrupted our global retail store operations",
    },
]
