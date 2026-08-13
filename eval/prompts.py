SINGLE_HOP_SYSTEM_PROMPT = """You are generating evaluation data for a RAG system.

You will be given the full text of one research paper. Write exactly 2 questions
that can each be answered using ONLY this paper.

Rules:
1. Questions must be answerable from a single, contiguous excerpt (1 to 3 sentences)
   in the paper. Do not write questions that require combining two far-apart sections.
2. Do not write questions whose answer is just the paper title or author names.
3. Do not write yes/no questions. Prefer "what", "how", "why", "which" questions
   that require pulling a specific fact, method, or result from the text.
4. The excerpt must be copied verbatim, character-for-character, from the paper.
   Do not paraphrase the excerpt.
5. The answer must be concise (1 to 2 sentences) and derived only from the excerpt.

Output format: return a single JSON object with key "samples", a list of exactly
2 objects, each with keys "question", "answer", "excerpt". No text outside the JSON.

Example 1:
Paper excerpt (input):
"...We evaluate our pruning method on ResNet-50 trained on ImageNet. At a 40%
sparsity level, our method achieves 76.1% top-1 accuracy, a 0.3 point drop from
the unpruned baseline of 76.4%..."

Output:
{
  "samples": [
    {
      "question": "What top-1 accuracy did the pruned ResNet-50 achieve at 40% sparsity?",
      "answer": "76.1% top-1 accuracy, a 0.3 point drop from the 76.4% unpruned baseline.",
      "excerpt": "At a 40% sparsity level, our method achieves 76.1% top-1 accuracy, a 0.3 point drop from the unpruned baseline of 76.4%"
    }
  ]
}

Example 2:
Paper excerpt (input):
"...Prior work on prompt injection defenses relies on input sanitization, which
fails against encoded payloads. We instead propose a runtime taint-tracking
approach that flags any model output influenced by untrusted input spans..."

Output:
{
  "samples": [
    {
      "question": "Why does input sanitization fail as a prompt injection defense according to the paper?",
      "answer": "Because it fails against encoded payloads.",
      "excerpt": "Prior work on prompt injection defenses relies on input sanitization, which fails against encoded payloads."
    }
  ]
}

Now generate exactly 2 questions for the paper provided by the user, following
this exact format and these exact rules."""


MULTI_HOP_SYSTEM_PROMPT = """You are generating evaluation data for a RAG system
that traverses a graph of semantically related chunks.

You will be given 2 or more excerpts, each from a different paper, that were
linked because they are semantically similar. You must work through this in
strict order. Do not skip steps.

STEP 0: Check if any excerpt is not real content.
Some excerpts may just be reference lists / bibliographies (patterns like
"[12] Author, A. Title. arXiv:1234.5678" repeated, or a block that is mostly
citation entries with no explanatory sentences). If ANY excerpt is a
reference list rather than actual discussion of ideas, set "valid_excerpts"
to false, and stop. Do not generate a question from citation-only content.

STEP 1: Summarize what each excerpt can answer alone.
For each excerpt, write one line: what specific fact or claim could a reader
get from THIS excerpt by itself, with no other excerpt. Be honest here, most
"related" chunks turn out to only support a single-excerpt question.

STEP 2: Decide if a genuine multi-hop question exists.
A genuine multi-hop question requires one specific fact from excerpt A AND
one specific fact from excerpt B to answer correctly, where neither fact
alone is sufficient. This is NOT the same as "both excerpts are about the
same topic." If you cannot point to one specific fact per excerpt that the
question needs, set "answerable_from_single_excerpt" to true, and leave
"question" and "answer" empty.

STEP 3: If a genuine multi-hop question exists, write it.
- The question must fail to be answerable if either excerpt is removed.
- Do not invent, infer, or guess facts not explicitly stated. If a value
  or detail is "not directly provided," do not write "can be inferred" —
  that is fabrication. Leave it out or discard the question instead.
- Do not mix facts from different papers together as if they belong to
  one source, and do not attribute a fact from excerpt A to the paper
  excerpt B came from.

STEP 4: Trace every claim in your answer.
For each sentence in your answer, note which excerpt (A, B, ...) it came
from. If a sentence can't be traced to a specific excerpt, delete it.

Output format: return a single JSON object with keys:
"valid_excerpts" (boolean),
"per_excerpt_summary" (list of strings, one per excerpt, from Step 1),
"answerable_from_single_excerpt" (boolean),
"question" (string, empty if not applicable),
"answer" (string, empty if not applicable),
"claim_trace" (list of {{"claim": str, "source_excerpt": "A"|"B"|...}}, empty if not applicable)
No text outside the JSON.

Example 1 (genuinely multi-hop, passes all steps):
Excerpt A (Paper: "Taint-Tracking Defenses for LLM Agents"):
"...our taint-tracking defense adds a median latency overhead of 340ms per agent
step, which the authors note may be prohibitive for real-time applications..."

Excerpt B (Paper: "Benchmarking Real-Time Constraints in Agentic Pipelines"):
"...production agentic pipelines used in customer-facing chat typically require
end-to-end response latency under 500ms to maintain acceptable user experience..."

Output:
{{
  "valid_excerpts": true,
  "per_excerpt_summary": [
    "Excerpt A alone: the taint-tracking defense adds 340ms latency per agent step.",
    "Excerpt B alone: customer-facing pipelines typically need under 500ms end-to-end latency."
  ],
  "answerable_from_single_excerpt": false,
  "question": "Would the taint-tracking defense's latency overhead be compatible with the response-time budget typically required for customer-facing agentic pipelines?",
  "answer": "Likely not on its own: the defense adds 340ms of latency per agent step, which would consume most of the under-500ms end-to-end budget that customer-facing pipelines typically require, leaving little room for the rest of the pipeline.",
  "claim_trace": [
    {{"claim": "the defense adds 340ms of latency per agent step", "source_excerpt": "A"}},
    {{"claim": "customer-facing pipelines typically require under-500ms end-to-end latency", "source_excerpt": "B"}}
  ]
}}

Example 2 (redundant, not multi-hop, fails at Step 2):
Excerpt A (Paper: "Survey of Prompt Injection Attacks"):
"...prompt injection attacks can be categorized into direct injection, where the
attacker controls the user prompt, and indirect injection, where malicious
instructions are embedded in retrieved documents..."

Excerpt B (Paper: "Indirect Prompt Injection via Tool Outputs"):
"...indirect prompt injection occurs when an attacker embeds instructions inside
content that the model later retrieves or processes, such as a webpage or document..."

Output:
{{
  "valid_excerpts": true,
  "per_excerpt_summary": [
    "Excerpt A alone: defines direct vs indirect prompt injection, including that indirect injection uses malicious instructions in retrieved documents.",
    "Excerpt B alone: explains indirect prompt injection occurs via retrieved content like webpages or documents — this is already covered by excerpt A."
  ],
  "answerable_from_single_excerpt": true,
  "question": "",
  "answer": "",
  "claim_trace": []
}}

Example 3 (bibliography-only excerpt, fails at Step 0):
Excerpt A: "[15] Hao Li, Ruoyao Wen, Shanghao Shi, Ning Zhang, and Chaowei Xiao.
Agentdyn: A dynamic open-ended benchmark for evaluating prompt injection attacks
of real-world agent security system. arXiv preprint arXiv:2602.03117, 2026."

Excerpt B: "[16] Milad Nasr, Nicholas Carlini... The attacker moves second:
Stronger adaptive attacks bypass defenses against llm jailbreaks and prompt
injections. arXiv preprint arXiv:2510.09023, 2025."

Output:
{{
  "valid_excerpts": false,
  "per_excerpt_summary": [],
  "answerable_from_single_excerpt": false,
  "question": "",
  "answer": "",
  "claim_trace": []
}}

Now evaluate the excerpts provided by the user, following this exact process
and format."""
