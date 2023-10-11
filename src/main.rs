use anyhow::Result;
use clap::Parser;
use opencv::{
    core::bitwise_and,
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
use std::{
    collections::HashSet,
    fs::{create_dir, read_dir, read_to_string},
    path::PathBuf,
};

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

fn perform_segmentation(
    input_filepath: PathBuf,
    output_filepath: PathBuf,
    mask_filepath: PathBuf,
    index: &Option<u8>,
    label: &Option<String>,
) -> Result<()> {
    let input_filepath = input_filepath.into_os_string().into_string().unwrap();
    let image = imread(&input_filepath, IMREAD_COLOR)?;

    let mask_filepath = mask_filepath.into_os_string().into_string().unwrap();
    let content = read_to_string(mask_filepath)?;
    let segments: Vec<Segment> = from_str(&content)?;

    let segment = match (index, label) {
        (Some(index), _) => &segments[*index as usize],
        (None, Some(label)) => {
            if let Some(segment) = segments
                .iter()
                .find(|segment| segment.label == label.as_str())
            {
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
    let mask = Mat::from_slice_rows_cols(bytes, image.rows() as usize, image.cols() as usize)?;

    // calling Mat::new_rows_cols causes memory corruption
    let mut segmented_image = {
        let bytes = vec![0u8; 0usize];
        Mat::from_slice_rows_cols(&bytes, 0, 0)?
    };

    bitwise_and(&image, &image, &mut segmented_image, &mask)?;

    let output_filepath = output_filepath.into_os_string().into_string().unwrap();
    imwrite(&output_filepath, &segmented_image, &Default::default())?;

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    match (
        args.input.is_dir(),
        args.mask.is_dir(),
        args.output.is_dir(),
        args.output.exists(),
    ) {
        (false, false, false, _) => {
            perform_segmentation(
                args.input.clone(),
                args.output.clone(),
                args.mask.clone(),
                &args.index,
                &args.label,
            )?;
        }
        (true, true, true, _) | (true, true, false, false) => {
            let input_dir = read_dir(args.input)?;
            let mask_dir = read_dir(&args.mask)?;

            let mut mask_file_stems = HashSet::new();
            for entry in mask_dir {
                let entry = entry?;
                let path = entry.path();
                let stem = path.file_stem().unwrap();
                assert!(path.as_path().extension().unwrap() == "json");
                mask_file_stems.insert(stem.to_owned());
            }

            let mut input_filepaths = Vec::new();
            for entry in input_dir {
                let entry = entry?;
                let path = entry.path();
                let stem = path.file_stem().unwrap();
                if mask_file_stems.contains(stem) {
                    input_filepaths.push(path);
                } else {
                    panic!(
                        "Mask not found for image \"{}\"",
                        entry.path().into_os_string().into_string().unwrap()
                    );
                }
            }

            if !args.output.exists() {
                create_dir(args.output.clone())?;
            }

            for input_filepath in input_filepaths.drain(..) {
                let input_path = input_filepath.as_path();
                let input_stem = input_path.file_stem().unwrap().to_str().unwrap();

                let output_filepath = args.output.clone().join(format!("{}.png", input_stem));
                let mask_filepath = args.mask.clone().join(format!("{}.json", input_stem));

                perform_segmentation(
                    input_filepath,
                    output_filepath,
                    mask_filepath,
                    &args.index,
                    &args.label,
                )?;
            }
        }
        _ => panic!("Invalid combination of input path, output, and mask paths."),
    }

    Ok(())
}
