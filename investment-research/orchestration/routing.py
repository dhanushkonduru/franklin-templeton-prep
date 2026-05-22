def review_router(state):

    if state.revision_count >= 2:
        return "end"

    review = state.critic_review.lower()

    if "revise" in review:
        return "report_writer"

    return "end"