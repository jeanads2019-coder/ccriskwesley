def build_credit_prompt(
    decision: str,
    prob: float,
    threshold: float,
    factors: list[dict],
    top_k: int = 2
) -> str:
    """
    Builds a natural language explanation prompt
    including possible improvement paths for the client.
    """

    # Ordena fatores por impacto absoluto (mais influentes)
    sorted_factors = sorted(
        factors,
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )

    main_factors = sorted_factors[:top_k]

    factors_text = ""
    for item in main_factors:
        factors_text += (
            f"- {item['feature']} (impact: {item['shap_value']:.3f})\n"
        )

    prompt = f"""
You are a credit risk analyst assistant.

Your task is to explain a credit decision using ONLY the information provided.
Do NOT mention models, SHAP, machine learning, scores, or probabilities explicitly.

Decision: {decision}

Main factors influencing the decision:
{factors_text}

Instructions:
1. First, clearly explain why the credit was {decision.lower()}, in simple language. Explain in two topics the main reason.
2. Then, suggest practical and realistic actions the client could take to improve
   their chances of approval in the future.
3. Each suggestion must be directly related to the factors listed above. Offer only 2 most important (based on the factors), short and direct suggestions.
4. Use a respectful, supportive, and customer-oriented tone.
5. Do NOT make promises of approval.

Write the explanation for a non-technical audience in Brazil Portuguese, and elaborate a short answer, to not be exaushausting for those who are reading. It need to be short and clear.
"""

    return prompt

