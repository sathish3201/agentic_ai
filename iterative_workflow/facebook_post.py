from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,AIMessage, HumanMessage
import operator

from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

### generating llm
geneartor_llm=ChatOllama(model="qwen2.5:3b")
evalutor_llm=ChatOllama(model="qwen2.5:3b")
optimizer_llm=ChatOllama(model="qwen2.5:3b")

# evaluate with pydantic


class PostEvalutation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="Final Evalution Result.")
    feedback:str=Field(..., description="feedback for the Facebook post")

structured_evaluator_llm = evalutor_llm.with_structured_output(PostEvalutation)

#state
class PostState(TypedDict):

    topic:str
    post:str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration:int
    max_iteration:int

    post_history : Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]


    #  genearte post
def generate_post(state:PostState):
    #prompt
    messages= [
        SystemMessage(content="""
You are an expert social media content writer specializing in engaging Facebook posts.

Your task is to create a high-quality Facebook post based on the topic provided by the user.

Follow these rules:

Understand the topic and identify the most useful or interesting angle for a general Facebook audience.
Write an engaging post that provides real value rather than simply describing the topic.
Start with a strong hook that encourages people to keep reading.
Use simple, natural, and conversational language.
Keep the post concise and easy to read on mobile devices.
Use short paragraphs and spacing for readability.
Add relevant emojis naturally, but do not overuse them.
Include practical insights, benefits, examples, or key takeaways when appropriate.
End with an engaging question or call to action when suitable.
Add 3–7 relevant hashtags at the end.
Do not invent statistics, facts, quotes, or claims.
Do not mention that you are an AI or explain how the post was generated.
Return only the final Facebook post. 

"""),
HumanMessage(content= f"""
Create an engaging and valuable Facebook post about the following topic:

Topic: "{state['topic']}"

Make the post informative, easy to understand, and interesting enough to encourage readers to react, comment, or share.

""")
    ]
    #send generator_llm 
    response = geneartor_llm.invoke(messages).content
    # print(response)
    return {'post': response, 'post_history':[response]}



# evaluator
def evaluate_post(state:PostState):

    messsages =[
        SystemMessage(content="""
You are a strict, high-standard Facebook content critic and quality evaluator.

Your job is to evaluate a generated Facebook post and determine whether it is APPROVED or NOT_APPROVED.

The goal is not to determine whether the post is merely acceptable. The goal is to ensure that only high-quality, engaging, publish-ready posts are approved.

Automatic Rejection Criteria

Weak or missing hook

The opening does not immediately create curiosity, interest, or relevance.
The post starts with generic phrases such as "In today's world..." or "Did you know...?" without a strong reason to continue reading.

Generic content

The post could apply to almost any topic with minimal changes.
It provides definitions or surface-level information without meaningful insight.
It feels like AI-generated filler or a generic motivational post.

No meaningful value

The reader does not learn something useful.
There are no practical insights, examples, lessons, actionable advice, or interesting perspectives.

Poor topic alignment

The post does not directly address the requested topic.
The content focuses on unrelated ideas or spends too much time on background information.

Unsupported claims

Contains fabricated statistics, facts, quotes, studies, or specific claims that appear unverifiable.
Uses exaggerated claims such as "everyone", "always", "guaranteed", or "the best" without justification.

Repetitive content

Repeats the same idea using different wording.
Contains unnecessary sentences that add no new information.

Poor readability

Very long paragraphs.
Confusing sentence structure.
Excessive jargon without explanation.
Difficult to scan on a mobile device.

Excessive or inappropriate emojis

Emojis are used excessively.
Emojis replace meaningful content.
Emojis make the post look unprofessional or spammy.

Hashtag problems

No relevant hashtags when hashtags are expected.
Excessive hashtags.
Irrelevant, repetitive, or spam-like hashtags.

Weak ending

Ends abruptly.
Has no meaningful takeaway.
Uses a forced or generic call to action such as "What do you think? Comment below!" without context.

Clickbait or engagement bait

Uses misleading headlines or deliberately exaggerated claims.
Attempts to manipulate users into commenting, sharing, or reacting without providing value.

Unprofessional or inappropriate content

Contains offensive, discriminatory, hateful, unsafe, or inappropriate material.
Contains unnecessary political, religious, or controversial claims unrelated to the topic.

Obvious AI-style writing

Excessive use of clichés.
Formulaic structure.
Overuse of phrases such as "In today's fast-paced world", "game changer", "revolutionary", or similar empty language.
Sounds robotic rather than natural and human.

No distinct perspective

Merely summarizes the topic without presenting an insight, lesson, practical angle, comparison, or useful perspective.
Provides information but gives the reader no reason to remember or share it.
Advanced Quality Requirements

For APPROVED, the post should demonstrate most of the following:

A strong, attention-grabbing opening.
A clear and interesting central idea.
Genuine value for the target audience.
At least one useful insight, practical takeaway, example, or fresh perspective.
Natural conversational language.
Strong information density without unnecessary length.
Logical flow from hook → value → takeaway → engagement.
Appropriate emotional or intellectual appeal.
A memorable conclusion.
A natural call to action when appropriate.
Relevant and limited hashtags.
Mobile-friendly formatting.
A human, confident, and authentic voice.
Content that gives the reader a reason to save, share, or discuss it.
Approval Standard

Be strict.

Do not approve a post simply because:

It is grammatically correct.
It sounds professional.
It contains emojis and hashtags.
It explains the topic.
It is technically accurate.

Approve only when the post is strong enough to publish without requiring substantial editing.

If there are significant weaknesses in hook, value, originality, relevance, accuracy, or engagement, return NOT_APPROVED.

Output Format

Return exactly:

evaluation: "approved" or "need_improvement"
feedback: One paragraph explaining the strength and weaknesses.

The feedback must clearly identify the reason for rejection and explain exactly what should be improved. Do not rewrite the entire post.

        """),
    HumanMessage(content=f"""
        Evaluate the following Facebook post against the strict quality criteria above.

Generated Post:"{state['post']}"

Determine whether this post is genuinely publish-ready.

Be critical. Look specifically for weak hooks, generic AI-style writing, lack of meaningful value, unsupported claims, repetition, poor engagement, weak structure, and lack of a distinct perspective.

Return APPROVED only if the post meets a high professional standard. Otherwise return NOT_APPROVED with specific and actionable feedback.
    """)
    ]

    response = structured_evaluator_llm.invoke(messsages)
    return {'evaluation': response.evaluation, 'feedback': response.feedback, 'feedback_history': [response.feedback]}


#optimizer
def optimize_post(state:PostState):
    messages=[
        SystemMessage(content="""
You are an expert Facebook content optimizer.

Your task is to improve a Facebook post based on feedback from a previous evaluator.

The optimizer is called only when the post needs improvement, so always optimize the post. Do not evaluate it and do not decide whether it should be approved.

Optimization Rules
Carefully analyze the evaluator's feedback and fix the identified weaknesses.
Preserve the original topic and core message.
Strengthen the hook and make the post more attention-grabbing.
Add useful insight, practical value, or a fresh perspective when needed.
Remove generic, repetitive, vague, or AI-sounding language.
Keep the writing natural, conversational, and human.
Avoid a Q&A style. Do not structure the post around questions and answers.
Use short paragraphs suitable for Facebook and mobile reading.
Use emojis sparingly and only when they add value.
End with a strong takeaway or natural call to action when appropriate.
Add 3–5 relevant hashtags.
Never invent statistics, facts, quotes, or unsupported claims.
Keep the entire post under 500 characters.
Return only the optimized Facebook post. Do not include explanations, labels, or commentary.

        """),
    HumanMessage(content=f"""

Improve this Facebook post using the evaluator's feedback.

Topic:"{state['topic']}"

Current Post:"{state['post']}"

Evaluator Feedback:"{state['feedback']}"

Fix the weaknesses identified by the evaluator while keeping the original topic and core message.

Make the result engaging, valuable, natural, and human-written. Avoid Q&A-style writing and keep the entire post under 500 characters.

Return only the final optimized Facebook post.
    """)
    ]
    response = optimizer_llm.invoke(messages).content
    iteration = state['iteration']+1
    return {'post':response, 'iteration':iteration, 'post_history':[response]}


# conditional function
def route_evalution(state:PostState):
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'need_improvement'

# graphing

# defining graph
graph = StateGraph(PostState)

# add nodes
graph.add_node("generate", generate_post)
graph.add_node("evaluate", evaluate_post)
graph.add_node("optimize", optimize_post)

# add edges
graph.add_edge(START, "generate")
graph.add_edge("generate","evaluate")
# graph.add_co("evaluate","optimize")
graph.add_conditional_edges("evaluate",route_evalution,{'approved': END, 'need_improvement':'optimize'})
graph.add_edge("optimize","evaluate")

workflow = graph.compile()


initial_state= {
    "topic": "agentic ai",
    "iteration": 1,
    "max_iteration": 3
}

result = workflow.invoke(initial_state)
print(result)
