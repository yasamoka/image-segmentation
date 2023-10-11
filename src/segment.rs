use serde::{Deserialize, Serialize};
use serde_with::{
    base64::{Base64, Standard},
    formats::Unpadded,
    serde_as,
};

#[serde_as]
#[derive(Serialize, Deserialize, Debug)]
pub struct Segment {
    // #[allow(dead_code)]
    // score: f32,
    pub label: String,
    #[serde_as(as = "Base64<Standard, Unpadded>")]
    pub mask: Vec<u8>,
}
