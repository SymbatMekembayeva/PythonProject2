import json
from typing import List, Dict, Any
from google import genai
import os
from services import rag_agent
from datetime import datetime

from services.rag_agent import RAGAgent


class LLMJudge:

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """
        Initialize LLM Judge.

        Args:
            api_key: Gemini API key
            model: Gemini model to use for judging
        """
        self.client = genai.Client(api_key="AIzaSyALlRTA7V5myO99ZCEuXF7MqTXMHYfx6mU")
        self.model = model

    def judge_answer(
            self,
            question: str,
            agent_answer: str,
            expected_answer: str
    ) -> Dict[str, Any]:
        """
        Judge if agent's answer matches the expected answer in meaning.

        Args:
            question: The question asked
            agent_answer: Answer from the RAG agent
            expected_answer: Ground truth answer

        Returns:
            Dictionary with judgment (True/False), reasoning, and confidence
        """
        # Create judgment prompt
        prompt = f"""You are an expert evaluator judging if two answers to the same question have the same meaning.

Question: {question}

Expected Answer (Ground Truth):
{expected_answer}

Agent's Answer:
{agent_answer}

Task: Determine if the Agent's Answer conveys the same core meaning as the Expected Answer.

Guidelines:
- Focus on semantic meaning, not exact wording
- The agent's answer should cover the main points of the expected answer
- Minor differences in phrasing or additional details are acceptable
- If the agent says "I don't have information" but the expected answer has content, that's WRONG
- If key facts or concepts are missing or incorrect, that's WRONG

Respond in this EXACT format:
JUDGMENT: [TRUE or FALSE]
CONFIDENCE: [0-100]
REASONING: [Brief explanation of your judgment]

Example responses:
JUDGMENT: TRUE
CONFIDENCE: 95
REASONING: Both answers correctly explain that RAG combines retrieval and generation, though worded differently.

JUDGMENT: FALSE
CONFIDENCE: 85
REASONING: The agent's answer is missing the key concept of attention mechanisms that is central to the expected answer.
"""

        try:
            # Call Gemini to judge
            response = self.client.models.generate_content(
                model=f"models/{self.model}",
                contents=prompt
            )

            judgment_text = response.text.strip()

            # Parse the response
            judgment = self._parse_judgment(judgment_text)

            return judgment

        except Exception as e:
            print(f"Error in judging: {e}")
            return {
                "match": False,
                "confidence": 0,
                "reasoning": f"Error during judgment: {e}",
                "raw_response": ""
            }

    def _parse_judgment(self, text: str) -> Dict[str, Any]:
        """
        Parse the judgment response from LLM.

        Args:
            text: Raw response from LLM

        Returns:
            Parsed judgment dictionary
        """
        lines = text.strip().split('\n')

        judgment = None
        confidence = 0
        reasoning = ""

        for line in lines:
            line = line.strip()

            if line.startswith("JUDGMENT:"):
                judgment_str = line.replace("JUDGMENT:", "").strip().upper()
                judgment = judgment_str == "TRUE"

            elif line.startswith("CONFIDENCE:"):
                confidence_str = line.replace("CONFIDENCE:", "").strip()
                try:
                    confidence = int(confidence_str)
                except:
                    confidence = 0

            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return {
            "match": judgment if judgment is not None else False,
            "confidence": confidence,
            "reasoning": reasoning,
            "raw_response": text
        }


class RAGEvaluator:
    """Evaluates RAG agent performance using LLM-as-Judge."""

    def __init__(
            self,
            rag_agent: rag_agent,
            judge: LLMJudge,
            dataset_path: str
    ):
        """
        Initialize evaluator.

        Args:
            rag_agent: The RAG agent to evaluate
            judge: LLM judge for comparing answers
            dataset_path: Path to JSON dataset file
        """
        self.rag_agent = rag_agent
        self.judge = judge
        self.dataset_path = dataset_path
        self.results = []

    def load_dataset(self) -> List[Dict[str, str]]:
        """
        Load evaluation dataset from JSON file.

        Returns:
            List of question-answer pairs
        """
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            print(f"✓ Loaded {len(dataset)} questions from {self.dataset_path}")
            return dataset
        except FileNotFoundError:
            print(f"❌ Dataset file not found: {self.dataset_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {e}")
            return []

    def evaluate(
            self,
            verbose: bool = True,
            save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation on the entire dataset.

        Args:
            verbose: Print detailed progress
            save_results: Save results to JSON file

        Returns:
            Evaluation summary with metrics
        """
        print("\n" + "=" * 80)
        print("STARTING RAG AGENT EVALUATION")
        print("=" * 80)

        # Load dataset
        dataset = self.load_dataset()
        if not dataset:
            return {"error": "No dataset loaded"}

        # Evaluate each question
        total = len(dataset)
        correct = 0
        total_confidence = 0

        for i, item in enumerate(dataset, 1):
            question = item['question']
            expected = item['expected_answer']

            if verbose:
                print(f"\n[{i}/{total}] Question: {question}")

            # Get agent's answer
            result = self.rag_agent.answer(
                query=question,
                stream=False,
                show_context=False
            )
            agent_answer = result['answer']

            if verbose:
                print(f"Agent's answer: {agent_answer[:100]}...")

            # Judge the answer
            judgment = self.judge.judge_answer(question, agent_answer, expected)

            # Store result
            eval_result = {
                "question": question,
                "expected_answer": expected,
                "agent_answer": agent_answer,
                "match": judgment['match'],
                "confidence": judgment['confidence'],
                "reasoning": judgment['reasoning'],
                "sources_used": result.get('num_sources', 0),
                "used_rag": result.get('used_rag', False)
            }
            self.results.append(eval_result)

            # Update metrics
            if judgment['match']:
                correct += 1
            total_confidence += judgment['confidence']

            # Print judgment
            status = "✓ CORRECT" if judgment['match'] else "✗ WRONG"
            if verbose:
                print(f"{status} (Confidence: {judgment['confidence']}%)")
                print(f"Reasoning: {judgment['reasoning']}")

        # Calculate summary
        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_confidence = total_confidence / total if total > 0 else 0

        summary = {
            "total_questions": total,
            "correct_answers": correct,
            "wrong_answers": total - correct,
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "timestamp": datetime.now().isoformat(),
            "detailed_results": self.results
        }

        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total Questions: {total}")
        print(f"Correct Answers: {correct}")
        print(f"Wrong Answers: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average Confidence: {avg_confidence:.2f}%")

        # Save results
        if save_results:
            self.save_results(summary)

        return summary

    def save_results(self, summary: Dict[str, Any]):
        """
        Save evaluation results to JSON file.

        Args:
            summary: Evaluation summary to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")

    def print_detailed_results(self):
        """Print detailed results for each question."""
        print("\n" + "=" * 80)
        print("DETAILED RESULTS")
        print("=" * 80)

        for i, result in enumerate(self.results, 1):
            print(f"\n[Question {i}]")
            print(f"Q: {result['question']}")
            print(f"\nExpected: {result['expected_answer'][:150]}...")
            print(f"\nAgent: {result['agent_answer'][:150]}...")
            print(f"\nMatch: {result['match']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Sources used: {result['sources_used']}")
            print("-" * 80)


def main():
    """Main function to run evaluation."""

    # Configuration
    DATASET_PATH = "../evaluation_dataset.json"
    GEMINI_API_KEY = "AIzaSyCEdbNx3pxK5cwD1fwggC-zgjpRMEHH_9U"
    COLLECTION_NAME = "pdf_documents"

    print("Initializing RAG Agent and LLM Judge...")

    # Initialize RAG Agent
    try:
        rag_agent = RAGAgent(
            collection_name=COLLECTION_NAME,
            gemini_api_key=GEMINI_API_KEY,
            gemini_model="gemini-1.5-flash",
            top_k=3,
            score_threshold=0.3
        )
    except Exception as e:
        print(f"❌ Error initializing RAG agent: {e}")
        return

    # Initialize LLM Judge
    judge = LLMJudge(
        api_key=GEMINI_API_KEY,
        model="gemini-1.5-flash"
    )

    # Initialize Evaluator
    evaluator = RAGEvaluator(
        rag_agent=rag_agent,
        judge=judge,
        dataset_path=DATASET_PATH
    )

    # Run evaluation
    summary = evaluator.evaluate(
        verbose=True,
        save_results=True
    )

    # Print detailed results
    print("\n" + "=" * 80)
    choice = input("\nShow detailed results for each question? (yes/no): ").strip().lower()
    if choice == 'yes':
        evaluator.print_detailed_results()


if __name__ == "__main__":
    main()