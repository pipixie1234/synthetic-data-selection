"""Exact candidate likelihoods while prefilling a shared prompt only once."""
import torch


@torch.inference_mode()
def candidate_values(model,prefixes,suffixes,pad_token_id,device):
    """Return flattened [prompt0/candidate0..3, prompt1/candidate0..3...] scores."""
    n_candidates=len(suffixes)
    length=max(map(len,prefixes))
    ids=torch.tensor([[pad_token_id]*(length-len(x))+x for x in prefixes],device=device)
    attention=torch.tensor([[0]*(length-len(x))+[1]*len(x) for x in prefixes],device=device)
    positions=(attention.cumsum(-1)-1).masked_fill(attention==0,0)
    output=model(input_ids=ids,attention_mask=attention,position_ids=positions,use_cache=True,logits_to_keep=1)
    first=torch.log_softmax(output.logits[:,-1].float(),dim=-1)
    first_tokens=torch.tensor([x[0] for x in suffixes],device=device)
    scores=first[:,first_tokens].reshape(-1)
    cache=output.past_key_values
    cache.batch_repeat_interleave(n_candidates)
    branch_attention=attention.repeat_interleave(n_candidates,dim=0)
    prefix_lengths=attention.sum(-1).repeat_interleave(n_candidates)
    expanded=suffixes*len(prefixes)
    for step in range(1,max(map(len,suffixes))):
        previous=torch.tensor([[suffix[step-1] if step-1<len(suffix) else pad_token_id] for suffix in expanded],device=device)
        branch_attention=torch.cat([branch_attention,torch.ones((len(expanded),1),dtype=branch_attention.dtype,device=device)],dim=-1)
        pos=(prefix_lengths+step-1)[:,None]
        output=model(input_ids=previous,attention_mask=branch_attention,position_ids=pos,past_key_values=cache,use_cache=True,logits_to_keep=1)
        cache=output.past_key_values
        logp=torch.log_softmax(output.logits[:,-1].float(),dim=-1)
        active=torch.tensor([step<len(x) for x in expanded],device=device)
        next_tokens=torch.tensor([x[step] if step<len(x) else pad_token_id for x in expanded],device=device)
        scores+=torch.where(active,logp[torch.arange(len(expanded),device=device),next_tokens],0)
    assert torch.isfinite(scores).all()
    return scores.double().cpu().tolist()
