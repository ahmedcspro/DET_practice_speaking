
import os
import time
from typing import Optional

import speech_recognition as sr
from google import genai
from google.genai import errors



MODEL_NAME = "gemini-3.6-flash"

API_KEY = os.getenv("GEMINI_API_KEY")

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2


SYSTEM_INSTRUCTION = """
You are an honest English tutor and DET preparation coach.

Your purpose is to help the student practice English for the
Duolingo English Test.

IMPORTANT:
You are NOT an official DET examiner and you must not claim
that your score is an official DET score.

============================================================
SESSION STRUCTURE
============================================================

At the beginning of the session:

1. Choose ONE clear topic.
2. Tell the student the topic.
3. Ask ONE clear speaking question.

Possible topics include:
- Education
- Technology
- Artificial intelligence
- Science
- Environment
- University
- Work
- Travel
- Social media
- Health
- Daily life

Keep the session focused on the SAME main topic.

Do NOT randomly change topics after every answer.

Instead, ask related follow-up questions that gradually
become more challenging.

Ask only ONE question at a time.

============================================================
QUESTION QUALITY
============================================================

Questions should encourage the student to:

- Give an opinion
- Explain reasons
- Give examples
- Describe experiences
- Compare ideas
- Discuss advantages and disadvantages
- Explain more complex ideas

============================================================
EVALUATION
============================================================

Evaluate the student's response based on:

1. Relevance to the question
2. Grammar
3. Vocabulary
4. Coherence
5. Development of ideas
6. Naturalness and fluency of the response

The application receives a speech-to-text transcription.

Do NOT pretend that the transcription alone can reliably
measure pronunciation.

Focus primarily on the language contained in the transcript.

Do not correct every tiny mistake.

Focus on the most important weaknesses that will help the
student improve.

============================================================
DET SCORE ESTIMATE
============================================================

Give an ESTIMATED DET SCORE RANGE rather than pretending
that one answer determines an exact score.

Use the DET scale from 10 to 160.

Use 5-point increments.

Examples:

90–100
105–115
120–130

The score is ONLY a practice estimate.

It is NOT an official DET score.

Never claim that the estimate is an official DET result.

If there is not enough evidence to make a meaningful estimate,
say:

"Not enough evidence yet."

As the session continues, use the student's overall
performance to make the estimated range more meaningful.

============================================================
FEEDBACK FORMAT
============================================================

After each answer, use exactly this structure:

DET ESTIMATED SCORE:
[estimated range or "Not enough evidence yet"]

TOPIC RELEVANCE:
[Pass or Fail]
[Brief explanation]

MAIN STRENGTH:
[One specific strength]

MAIN WEAKNESS:
[One important weakness]

GRAMMAR & COHESION:
[Important corrections and explanations]

VOCABULARY UPGRADES:
1. [weak/simple word] → [better alternative]
2. [weak/simple word] → [better alternative]

FLUENCY:
[Brief feedback about sentence flow, repetition,
hesitation-related wording, or unnatural phrasing
visible in the transcript]

MODEL RESPONSE:
[Give a natural strong response using the student's
original idea. Do not change the student's main idea.]

NEXT QUESTION:
[One new question related to the SAME topic]

Keep feedback concise and useful.

============================================================
OFF-TOPIC ANSWERS
============================================================

If the student does not answer the question:

- Mark Topic Relevance as Fail.
- Explain briefly why.
- Encourage the student to answer the actual question.
- Do not invent grammar or vocabulary mistakes that
  are not present.

============================================================
SPECIAL COMMANDS
============================================================

If the student types:

skip
example
I don't know
don't know

Do NOT penalize the student.

Instead:

1. Give a strong model answer to the CURRENT question.
2. Explain two useful expressions or structures.
3. Ask ONE new question about the SAME topic.

Do not change the topic.

============================================================
SESSION REVIEW
============================================================

If the student types:

review

Provide:

SESSION REVIEW

MAIN TOPIC:
...

ESTIMATED DET RANGE:
...

MAIN STRENGTH:
...

MAIN WEAKNESS:
...

TOP 3 RECURRING MISTAKES:
1.
2.
3.

VOCABULARY TO IMPROVE:
1.
2.

GRAMMAR TO PRACTICE:
1.
2.

NEXT GOAL:
...

Keep the review concise and practical.
"""
class DETVoiceTutor:
    """Voice-based English practice tutor for DET preparation."""

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

        self.client: Optional[genai.Client] = None
        self.chat = None

        self.recognizer = sr.Recognizer()

    def initialize(self) -> bool:
        """Initialize the Gemini client and chat session."""

        if not self.api_key:
            print("\n GEMINI_API_KEY is not set.")
            print("\nWindows PowerShell:")
            print('$env:GEMINI_API_KEY="YOUR_API_KEY"')
            return False

        try:
            self.client = genai.Client(api_key=self.api_key)

            self.chat = self.client.chats.create(
                model=self.model_name,
                config={
                    "system_instruction": SYSTEM_INSTRUCTION
                },
            )

            return True

        except Exception as error:
            print(f"\n Initialization error: {error}")
            return False


    def send_message(self, message: str):
        """
        Send a message to Gemini.

        Temporary 429, 500 and 503 errors are retried
        automatically with exponential backoff.
        """

        delay = INITIAL_RETRY_DELAY

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self.chat.send_message(message=message)

            except errors.APIError as error:

                if (
                    error.code in (429, 500, 503)
                    and attempt < MAX_RETRIES
                ):
                    print(
                        "\n Gemini is temporarily unavailable. "
                        f"Retrying in {delay} seconds..."
                    )

                    time.sleep(delay)
                    delay *= 2

                else:
                    raise

        return None

    # --------------------------------------------------------========= 

    def listen_to_microphone(self) -> Optional[str]:
        """Record the student's speech and convert it to text."""

        try:
            with sr.Microphone() as source:

                print("\n🎤 Microphone is OPEN.")
                print("Start speaking...")

                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=0.5,
                )

                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=10,
                        phrase_time_limit=30,
                    )

                except sr.WaitTimeoutError:
                    print(
                        "\n You did not start speaking "
                        "within 10 seconds."
                    )
                    return None

        except OSError as error:
            print("\n Microphone error.")
            print(f"Details: {error}")
            return None

        try:
            print("\n🤖 Converting speech to text...")

            text = self.recognizer.recognize_google(
                audio,
                language="en-US",
            )

            print(f'\n👉 You said: "{text}"')

            return text

        except sr.UnknownValueError:
            print(
                "\n❌ I could not understand the recording."
            )
            return None

        except sr.RequestError as error:
            print(
                "\n❌ Speech recognition service error."
            )
            print(f"Details: {error}")
            return None


    def start_session(self) -> bool:
        """Generate the first topic and question."""

        prompt = """
Start a new DET English speaking practice session.

Choose ONE realistic topic.

Display:

📝 MAIN TOPIC: [topic]

Then ask ONE clear DET-style speaking question.

Do not give feedback yet.

Keep the topic focused so that future questions can remain
within the same topic.
"""

        try:
            response = self.send_message(prompt)

            if response is None:
                return False

            print(f"\n🤖 Tutor:\n{response.text}")
            return True

        except errors.APIError as error:
            print(f"\n❌ Could not start the session: {error}")
            return False


    def evaluate_answer(self, answer: str) -> None:
        """Evaluate the student's transcribed spoken answer."""

        prompt = f"""
Evaluate the student's spoken English response.

CURRENT STUDENT RESPONSE:
"{answer}"

Follow the evaluation format from your system instructions.

Important:

- Stay on the SAME main topic.
- Check whether the answer directly addresses the question.
- Give an estimated DET SCORE RANGE.
- Do not claim the score is official.
- Identify one main strength.
- Identify one main weakness.
- Give useful grammar and cohesion feedback.
- Give two useful vocabulary upgrades.
- Comment on fluency based only on the transcript.
- Do not pretend that pronunciation can be accurately
  diagnosed from text alone.
- Give one natural model response using the student's idea.
- Ask ONE related follow-up question.
- Keep the response concise.
"""

        try:
            response = self.send_message(prompt)

            if response is None:
                return

            print("\n" + "=" * 60)
            print(response.text)
            print("=" * 60)

        except errors.APIError as error:
            print(f"\n Gemini API error: {error}")

    def show_example(self) -> None:
        """Show a model answer without penalizing the student."""

        prompt = """
The student requested an example.

Do NOT penalize the student.

For the CURRENT question:

1. Give a strong, natural model answer.
2. Keep it realistic for a strong DET response.
3. Explain two useful expressions or structures.
4. Ask ONE new question related to the SAME topic.

Do not change the topic.
"""

        try:
            response = self.send_message(prompt)

            if response is None:
                return

            print(f"\n🤖 Model Answer:\n{response.text}")

        except errors.APIError as error:
            print(f"\n❌ Gemini API error: {error}")


    def review_session(self) -> None:
        """Review the student's performance."""

        prompt = """
Review my English performance throughout this session.

Use this structure:

SESSION REVIEW

MAIN TOPIC:
...

ESTIMATED DET RANGE:
...

MAIN STRENGTH:
...

MAIN WEAKNESS:
...

TOP 3 RECURRING MISTAKES:
1.
2.
3.

VOCABULARY TO IMPROVE:
1.
2.

GRAMMAR TO PRACTICE:
1.
2.

NEXT GOAL:
...

Be honest, concise and practical.
"""

        try:
            response = self.send_message(prompt)

            if response is None:
                return

            print(f"\n📊 {response.text}")

        except errors.APIError as error:
            print(f"\n❌ Gemini API error: {error}")

    @staticmethod
    def show_help() -> None:
        """Display available commands."""

        print(
            """
COMMANDS
--------
help       Show available commands
skip       Get a model answer
example    Get a model answer
review     Review your session
quit       Exit
exit       Exit
"""
        )


    def run(self) -> None:
        """Run the interactive DET speaking tutor."""

        if not self.initialize():
            return

        print("\n" + "=" * 60)
        print("          DET ENGLISH SPEAKING PRACTICE")
        print("=" * 60)
        print("• Main-topic focused practice")
        print("• DET estimated score range")
        print("• Grammar and vocabulary feedback")
        print("• Topic relevance checking")
        print("• Voice-to-text practice")
        print("• Session review")
        print("• Type 'skip' or 'example' for help")
        print("• Type 'review' for a session summary")
        print("• Type 'help' for commands")
        print("• Type 'quit' to exit")
        print("=" * 60)

        if not self.start_session():
            return

        while True:

            try:
                user_action = input(
                    "\n[Press Enter to speak, "
                    "or type skip / example / review / quit]: "
                ).strip()

                command = user_action.lower()


                if command in {"quit", "exit"}:
                    print("\nGoodbye! Keep practicing! ")
                    break

                if command == "help":
                    self.show_help()
                    continue


                if command == "review":
                    print("\n📊 Reviewing your session...")
                    self.review_session()
                    continue

                if command in {
                    "skip",
                    "example",
                    "i don't know",
                    "dont know",
                    "don't know",
                }:
                    print(
                        "\n🤖 Generating a model answer..."
                    )
                    self.show_example()
                    continue


                print("\n🎤 Preparing microphone...")

                spoken_text = self.listen_to_microphone()

                if spoken_text:

                    print(
                        "\n🤖 Evaluating your spoken response..."
                    )

                    self.evaluate_answer(spoken_text)

            except errors.APIError as error:

                print(f"\n Gemini API error: {error}")

                if error.code == 503:
                    print(
                        "Gemini is temporarily unavailable. "
                        "Please wait and try again."
                    )

            except KeyboardInterrupt:

                print("\n\nGoodbye! ")
                break

            except Exception as error:

                print(
                    f"\n Unexpected error: {error}"
                )

if __name__ == "__main__":

    tutor = DETVoiceTutor(
        api_key=API_KEY,
        model_name=MODEL_NAME,
    )

    tutor.run()