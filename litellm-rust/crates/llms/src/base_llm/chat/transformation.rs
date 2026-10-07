use litellm_llms_types::formats::chat_completions::{
    ChatCompletionsResponse, ChatMessage, ChatMessageContent,
};
use serde_json::{Map, Value};

use crate::{
    Error,
    base_llm::chat::streaming::{ChatStream, StreamShape},
};

/// The provider-shaped request body a config produces. Named rather than a bare
/// `Value` so the transform contract stays a typed one, mirroring
/// [`crate::base_llm::audio_transcription::transformation::AudioTranscriptionRequestData`].
pub struct ProviderChatRequestData {
    pub body: Value,
    pub stream_shape: StreamShape,
}

/// The raw provider response body handed back to a config for normalization.
pub struct ProviderChatResponseData {
    pub body: Value,
}

pub const STREAM_PARAM: &str = "stream";

/// Message fields that carry no meaning for the upstream body, so their
/// presence does not make a request untranslatable.
const IGNORABLE_MESSAGE_FIELDS: &[&str] = &["name"];

pub use crate::base_llm::auth::{Headers, ValidatedEnvironment};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unsupported(pub &'static str);

pub trait BaseConfig: Sync {
    fn secret_names(&self) -> Vec<&'static str>;

    /// Supported OpenAI parameter names paired with their provider names.
    fn supported_openai_param_mappings(&self) -> &'static [(&'static str, &'static str)];

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
    ) -> Result<ProviderChatRequestData, Error>;

    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
    ) -> Result<ChatCompletionsResponse, Error>;

    fn model_response_iterator(&self, _shape: StreamShape) -> Option<ChatStream> {
        None
    }

    /// Shapes the forwarded headers and names the credential, the way Python's
    /// `validate_environment` does, without applying it: `resolve_auth` does that once
    /// for every config.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error>;

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    /// Parameters consumed as call configuration (credentials, endpoints)
    /// rather than placed in the body. Accepted, never serialized.
    fn config_params(&self) -> &'static [&'static str] {
        &[]
    }

    fn unsupported_reason(
        &self,
        messages: &[ChatMessage],
        optional_params: &Map<String, Value>,
    ) -> Option<Unsupported> {
        unsupported_stream(optional_params)
            .or_else(|| messages.iter().find_map(unsupported_message))
    }
}

pub fn unsupported_stream(optional_params: &Map<String, Value>) -> Option<Unsupported> {
    if optional_params
        .get(STREAM_PARAM)
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Some(Unsupported("streaming"));
    }
    None
}

pub fn unsupported_message(message: &ChatMessage) -> Option<Unsupported> {
    if message
        .extra
        .keys()
        .any(|key| !IGNORABLE_MESSAGE_FIELDS.contains(&key.as_str()))
    {
        return Some(Unsupported("unrecognized message field"));
    }
    if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
        return Some(Unsupported("unrecognized message role"));
    }
    match &message.content {
        None => Some(Unsupported("message without content")),
        Some(ChatMessageContent::Text(_)) => None,
        Some(ChatMessageContent::Parts(parts)) if parts.is_empty() => {
            Some(Unsupported("message without content"))
        }
        Some(ChatMessageContent::Parts(parts)) => parts
            .iter()
            .any(|part| {
                part.get("type").and_then(Value::as_str) != Some("text")
                    || part.get("text").and_then(Value::as_str).is_none()
                    || part.as_object().is_some_and(|object| object.len() != 2)
            })
            .then_some(Unsupported("non-text message content")),
    }
}
