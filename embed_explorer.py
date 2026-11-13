import torch
import argparse
from model_analyze import load_model, get_vector

def main():
    parser = argparse.ArgumentParser(description="Pick a word, look at it's token embedding")
    parser.add_argument("--word", type=str, default="Shell", help="Choose a word")
    parser.add_argument("--minus", type=str, default="Sea", help="Take a vector from the first word")

    args = parser.parse_args()

    model, tokenizer = load_model("HuggingFaceTB/SmolLM2-135M")

    embed_matrix = model.model.embed_tokens.weight.detach()
    
    word = args.word
    minusword = args.minus

    word_vec_embed = get_vector(word, tokenizer, embed_matrix)
    minus_vec_embed = get_vector(minusword, tokenizer, embed_matrix)

    token_id = tokenizer.encode(word, add_special_tokens=False)[0]
    vec = embed_matrix[token_id]
    
    print(f"Token: {repr(word)} | ID {token_id}")
    print(f"First 10 of 576 Coordinates: {vec[:10].tolist()}")
    print(f"Mean Value (Center is 0): {vec.mean().item():.4f}")
    print(f"Vector Magnitude (weight/length): {vec.norm().item():.4f}")

    norm_matrix = embed_matrix / embed_matrix.norm(dim=1, keepdim=True)
    norm_vec = vec / vec.norm()
    similarities = torch.matmul(norm_matrix, norm_vec)

    word_vec_norm = get_vector(word, tokenizer, norm_matrix)
    minus_vec_norm = get_vector(minusword, tokenizer, norm_matrix)

    top_scores, top_ids = torch.topk(similarities, 12)
    print("\nClosest words in raw embedding space:")
    for score, idx in zip(top_scores, top_ids):
        print(f"{score.item():+.4f} | ID {idx.item():5d} | {repr(tokenizer.decode([idx.item()]))}")

    print(f"And now we try to take away {minusword} from {word}")

    embed_target_vec = word_vec_embed - minus_vec_embed
    norm_target_vec = word_vec_norm - minus_vec_norm

    embed_target_vec = embed_target_vec / embed_target_vec.norm()
    norm_target_vec = norm_target_vec / norm_target_vec.norm()
    
    embed_similarities = torch.matmul(norm_matrix, embed_target_vec)
    norm_similarities = torch.matmul(norm_matrix, norm_target_vec)

    embed_top_score, embed_top_ids = torch.topk(embed_similarities, 10)
    norm_top_score, norm_top_ids = torch.topk(norm_similarities, 10)

    print(f"This is what the embedded looked like:")
    for embed_top, embed_id in zip(embed_top_score, embed_top_ids):
        print(f"{embed_top.item():+.4f} | ID {embed_id.item():5d} | {repr(tokenizer.decode([embed_id.item()]))}")

    print(f"This is what the normed looked like:")
    for norm_top, norm_id in zip(norm_top_score, norm_top_ids):
        print(f"{norm_top.item():+.4f} | ID {norm_id.item():5d} | {repr(tokenizer.decode([norm_id.item()]))}")


if __name__ == "__main__":
    main()

