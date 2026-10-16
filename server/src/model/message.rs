//! Defines canonical append-only Conversation Messages.
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ToolImageReference, ToolRoundId};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    Prompt,
    User,
    Runtime,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCallContent {
    pub index: usize,
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolResultContent {
    pub call_id: String,
    pub name: String,
    pub content: String,
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<ToolImageReference>,
    #[serde(skip)]
    pub provider_parts: Vec<ContentPart>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProviderReplayState {
    pub provider_kind: String,
    pub value: Value,
}

impl ProviderReplayState {
    /// Fold another signature from the same assistant turn into this one.
    /// Identical states are kept once. Same-provider blocks are concatenated
    /// in arrival order so a later Cursor turn can replay every thinking block.
    pub fn merge(self, other: Self) -> Result<Self, &'static str> {
        if self == other {
            return Ok(self);
        }
        if self.provider_kind != other.provider_kind {
            if self.provider_kind == "cursor_opaque" {
                return Ok(other);
            }
            if other.provider_kind == "cursor_opaque" {
                return Ok(self);
            }
            return Err("provider replay states disagree");
        }
        let value = match self.provider_kind.as_str() {
            "anthropic" => concat_replay_array(&self.value, &other.value, "blocks"),
            "openai_responses" => concat_replay_array(&self.value, &other.value, "items"),
            "openai_chat" => {
                let reasoning = format!(
                    "{}{}",
                    replay_text(&self.value, "reasoning_content"),
                    replay_text(&other.value, "reasoning_content")
                );
                serde_json::json!({"reasoning_content": reasoning})
            }
            "cursor_opaque" => {
                let mut parts = opaque_replay_parts(&self.value);
                extend_opaque_parts(&mut parts, opaque_replay_parts(&other.value));
                serde_json::json!({"parts": parts})
            }
            _ => return Err("provider replay states disagree"),
        };
        Ok(Self {
            provider_kind: self.provider_kind,
            value,
        })
    }

    pub fn cursor_reasoning_parts(&self) -> Option<Vec<Value>> {
        if self.provider_kind != "cursor_opaque" {
            return None;
        }
        self.value.get("parts").and_then(Value::as_array).cloned()
    }
}

fn replay_text(value: &Value, field: &str) -> String {
    value
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn concat_replay_array(left: &Value, right: &Value, field: &str) -> Value {
    let mut items = left
        .get(field)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if let Some(extra) = right.get(field).and_then(Value::as_array) {
        items.extend(extra.iter().cloned());
    }
    serde_json::json!({ field: items })
}

fn opaque_replay_parts(value: &Value) -> Vec<Value> {
    match value {
        Value::String(signature) => vec![serde_json::json!({"signature": signature})],
        _ => value
            .get("parts")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
    }
}

fn extend_opaque_parts(parts: &mut Vec<Value>, extra: Vec<Value>) {
    for part in extra {
        let signature = part
            .get("signature")
            .and_then(Value::as_str)
            .map(str::to_string);
        let text = part
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if let Some(object) = parts
            .last_mut()
            .and_then(Value::as_object_mut)
            .filter(|last| last.get("signature").and_then(Value::as_str) == signature.as_deref())
        {
            let combined = format!(
                "{}{text}",
                object
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            );
            object.insert("text".into(), Value::String(combined));
            continue;
        }
        parts.push(part);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        mime_type: String,
        #[serde(with = "base64_bytes")]
        data: Vec<u8>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MessageContent {
    Parts {
        parts: Vec<ContentPart>,
    },
    Assistant {
        text: String,
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_round_id: Option<ToolRoundId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        replay_state: Option<ProviderReplayState>,
        tool_calls: Vec<ToolCallContent>,
    },
    ToolResult(ToolResultContent),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CanonicalMessage {
    pub message_id: String,
    pub role: Role,
    pub origin: Origin,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_event_id: Option<String>,
}

impl CanonicalMessage {
    pub fn text(
        message_id: impl Into<String>,
        role: Role,
        origin: Origin,
        text: impl Into<String>,
    ) -> Self {
        Self {
            message_id: message_id.into(),
            role,
            origin,
            content: MessageContent::Parts {
                parts: vec![ContentPart::Text { text: text.into() }],
            },
            runtime_event_id: None,
        }
    }

    pub fn parts(
        message_id: impl Into<String>,
        role: Role,
        origin: Origin,
        parts: Vec<ContentPart>,
    ) -> Self {
        Self {
            message_id: message_id.into(),
            role,
            origin,
            content: MessageContent::Parts { parts },
            runtime_event_id: None,
        }
    }
}

mod base64_bytes {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(data: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&STANDARD.encode(data))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        STANDARD.decode(encoded).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeEvent {
    pub event_id: String,
    pub text: String,
}

impl RuntimeEvent {
    pub fn into_message(self) -> CanonicalMessage {
        CanonicalMessage {
            message_id: format!("runtime:{}", self.event_id),
            role: Role::User,
            origin: Origin::Runtime,
            content: MessageContent::Parts {
                parts: vec![ContentPart::Text { text: self.text }],
            },
            runtime_event_id: Some(self.event_id),
        }
    }
}
