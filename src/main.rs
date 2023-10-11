use anyhow::Result;
use clap::Parser;
use opencv::{
    core::{bitwise_and, CV_8UC3},
    imgcodecs::{imread, imwrite, IMREAD_COLOR},
    prelude::*,
};
use png::Decoder;
use serde::Deserialize;
use serde_json::from_str;
use serde_with::{
    base64::{Base64, Standard},
    formats::Unpadded,
    serde_as,
};
use std::{fs::read_to_string, path::PathBuf};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(short, long)]
    mask: PathBuf,

    #[arg(long)]
    index: Option<u8>,

    #[arg(long)]
    label: Option<String>,
}

#[serde_as]
#[derive(Deserialize, Debug)]
struct Segment {
    #[allow(dead_code)]
    score: f32,
    label: String,
    #[serde_as(as = "Base64<Standard, Unpadded>")]
    mask: Vec<u8>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let input_filepath = args.input.into_os_string().into_string().unwrap();
    let output_filepath = args.output.into_os_string().into_string().unwrap();
    let mask_filepath = args.mask.into_os_string().into_string().unwrap();

    let image = imread(&input_filepath, IMREAD_COLOR)?;

    let content = read_to_string(mask_filepath)?;
    let segments: Vec<Segment> = from_str(&content)?;

    let segment = match (args.index, args.label) {
        (Some(index), _) => &segments[index as usize],
        (None, Some(label)) => {
            if let Some(segment) = segments.iter().find(|segment| segment.label == label) {
                segment
            } else {
                panic!("Label was not found.")
            }
        }
        (None, None) => {
            panic!("Either index or label must be provided.");
        }
    };

    let decoder = Decoder::new(segment.mask.as_slice());
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let bytes = &buf[..info.buffer_size()];

    let mask = Mat::from_slice(bytes)?;
    let mask = mask.reshape(1, image.rows())?;
    let mut segmented_image = unsafe { Mat::new_rows_cols(image.rows(), image.cols(), CV_8UC3) }?;
    bitwise_and(&image, &image, &mut segmented_image, &mask)?;

    imwrite(&output_filepath, &segmented_image, &Default::default())?;

    Ok(())
}
