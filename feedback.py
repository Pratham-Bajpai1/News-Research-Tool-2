import json
import os
import streamlit as st
from datetime import datetime
from typing import List, Dict
import pandas as pd

# Constants
FEEDBACK_FILE = "data/feedbacks.json"
os.makedirs("data", exist_ok=True)


class FeedbackSystem:
    def __init__(self):
        self.feedbacks = self._load_feedbacks()

    def _load_feedbacks(self) -> List[Dict]:
        """Load existing feedbacks from JSON file"""
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def _save_feedbacks(self):
        """Save feedbacks to JSON file"""
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(self.feedbacks, f, ensure_ascii=False, indent=2)

    def add_feedback(self, rating: int, comment: str, user_id: str = "anonymous"):
        """Add new feedback to the system"""
        new_feedback = {
            "id": len(self.feedbacks) + 1,
            "user_id": user_id,
            "rating": rating,
            "comment": comment.strip(),
            "timestamp": datetime.now().isoformat(),
            "upvotes": 0
        }
        self.feedbacks.append(new_feedback)
        self._save_feedbacks()
        return new_feedback

    def upvote_feedback(self, feedback_id: int):
        """Increment upvote count for a feedback"""
        for feedback in self.feedbacks:
            if feedback["id"] == feedback_id:
                feedback["upvotes"] += 1
                self._save_feedbacks()
                return True
        return False

    def get_feedbacks(self, sort_by: str = "newest") -> List[Dict]:
        """Get all feedbacks with sorting options"""
        feedbacks = self.feedbacks.copy()
        if sort_by == "newest":
            return sorted(feedbacks, key=lambda x: x["timestamp"], reverse=True)
        elif sort_by == "top_rated":
            return sorted(feedbacks, key=lambda x: (x["rating"], x["upvotes"]), reverse=True)
        elif sort_by == "most_upvoted":
            return sorted(feedbacks, key=lambda x: x["upvotes"], reverse=True)
        return feedbacks


def show_feedback_page():
    """Main feedback interface"""
    feedback_system = FeedbackSystem()

    st.title("📝 Feedback & Ratings")
    st.markdown("Help us improve by sharing your experience with the tool")

    # Feedback submission form
    with st.form("feedback_form"):
        st.subheader("Share Your Feedback")

        col1, col2 = st.columns([1, 3])
        with col1:
            rating = st.selectbox(
                "Rating (1-5 stars)",
                options=[5, 4, 3, 2, 1],
                format_func=lambda x: "⭐" * x,
                index=0
            )
        with col2:
            user_id = st.text_input("Your Name (optional)", "anonymous")

        comment = st.text_area("Your Feedback",
                               placeholder="What did you like? What can we improve?",
                               height=150)

        submitted = st.form_submit_button("Submit Feedback")

        if submitted and comment.strip():
            feedback_system.add_feedback(rating, comment, user_id)
            st.success("Thank you for your feedback! 💖")

    # Feedback display section
    st.markdown("---")
    st.subheader("Community Feedback")

    # Sorting options
    sort_option = st.selectbox(
        "Sort by",
        ["Newest First", "Top Rated", "Most Upvoted"],
        index=0
    )

    sort_mapping = {
        "Newest First": "newest",
        "Top Rated": "top_rated",
        "Most Upvoted": "most_upvoted"
    }

    feedbacks = feedback_system.get_feedbacks(sort_mapping[sort_option])

    if not feedbacks:
        st.info("No feedback yet. Be the first to share!")
    else:
        for feedback in feedbacks:
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"### {'⭐' * feedback['rating']}")
                    st.caption(feedback['timestamp'][:10])
                    st.markdown(f"**{feedback['user_id']}**")
                with col2:
                    st.markdown(f"#### {feedback['comment']}")
                    if st.button("👍 Upvote", key=f"upvote_{feedback['id']}"):
                        feedback_system.upvote_feedback(feedback['id'])
                        st.rerun()
                    st.caption(f"{feedback['upvotes']} upvotes")

                st.markdown("---")