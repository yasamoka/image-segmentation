use std::{
    fs::{create_dir, read_dir},
    path::PathBuf,
};

use anyhow::Result;
use clap::Parser;
// use futures::future::try_join_all;
use reqwest::Client;
use segment::Segment;
use serde_json::{from_str, to_string};
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncWriteExt, BufReader},
};

const ENDPOINT: &str = "https://api-inference.huggingface.co/models";

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(short, long)]
    model: String,

    #[arg(short, long)]
    token: String,
}

async fn infer(
    model: String,
    token: String,
    image_filepath: PathBuf,
    mask_filepath: PathBuf,
) -> Result<()> {
    let payload = {
        let file = File::open(image_filepath).await?;
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).await?;
        buffer
    };

    let client = Client::new();
    let url = format!("{}/{}", ENDPOINT, model);
    let res = client
        .post(url)
        .body(payload)
        .bearer_auth(token)
        .send()
        .await?;

    assert!(res.status() == 200);

    let text = res.text().await?;
    let segments = from_str::<Vec<Segment>>(&text)?;

    {
        let mut file = File::create(mask_filepath).await?;
        let segment = to_string(&segments)?;
        let buf = segment.as_bytes();
        file.write_all(buf).await?;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    match (
        args.input.is_dir(),
        args.output.is_dir(),
        args.output.exists(),
    ) {
        (false, false, _) => {
            infer(args.model, args.token, args.input, args.output).await?;
        }
        (true, false, false) | (true, true, true) => {
            if !args.output.exists() {
                create_dir(args.output.clone())?;
            }

            let image_dir = read_dir(args.input)?;

            let futures = image_dir
                .map(|entry| {
                    let entry = entry?;
                    let image_filepath = entry.path();
                    let image_stem = image_filepath.file_stem().unwrap().to_str().unwrap();
                    let mask_filepath = args.output.clone().join(format!("{}.json", image_stem));
                    Ok(infer(
                        args.model.clone(),
                        args.token.clone(),
                        image_filepath,
                        mask_filepath,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;

            // let handles = futures
            //     .drain(..)
            //     .map(|future| tokio::spawn(future))
            //     .collect::<Vec<_>>();

            // try_join_all(handles).await?;

            // try_join_all(futures).await?;

            for future in futures {
                future.await?;
            }
        }
        _ => panic!("Invalid combination of input and output paths."),
    }

    Ok(())
}
