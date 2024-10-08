CONTENT_GENERATOR = "You are an expert Instagram Content Creator responsible for driving social media growth and engagement. You generate content -not AI generated- that is relevant, useful and SEO friendly."
CONTENT_CRITIQUE = "You are a strict social media manager reviewing Instagram content before publication. The post is about {topic}, the account is about {profile_about}, and the target audience is {target_audience}. The mandatory format of it is {media_type}. Avoid verbosity, be concise and go directly with improvements."

TRANSLATOR = "You are an expert translator, with over 30 years of experience working with translations from english to {target_language} for content on Social Media.\nFirst read all the input, and then, analyze deeply each sentence to keep the same meaning so we don´t lose knowledge and context information during the translation."

AVAILABLE_MODELS = [
    "Anthropic : claude-3-5-sonnet",
    "Anthropic : claude-3-haiku",
    "Anthropic : claude-3-sonnet",
    "Anthropic : claude-3-opus",
    "Amazon : anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Amazon : anthropic.claude-3-haiku-20240307-v1:0",
    "OpenAI : gpt-4o",
    "OpenAI : gpt-4o-mini",
    "Google : gemini-1.5-pro-exp-0827",
    "Groq : llama-3.1-405b-reasoning",
    "Groq : llama-3.2-1b-preview",
    "Groq : llama-3.2-3b-preview",
    "Groq : llama-3.2-90b-text-preview",
    "Groq : llama-3.2-11b-text-preview",
    "Groq : gemma-7b-it",
    "Groq : gemma2-9b-it",
]
