/* Frozen independent Transformers 5.15.0 F32/eager reference, written by tests/qwen35_fixture_gen.py visual_text.
 * E8 F12 H4 KV2 HD8 RoPE8; explicit 3-axis positions and embedding rows.
 * Parameters: ((arange(n)*7 + parameter_index*3)%23-11)*0.0175f;
 * linear_attn.norm adds 1; the writer computes them, only the outputs are listed. Tokens: 0,3,5,12,2,6,7. */
typedef struct {const char *name;uint32_t ndim;uint64_t dims[3];int param,plus_one,drop;} visual_text_tensor;
static const visual_text_tensor visual_text_tensors[]={
{"lm_head.weight",2,{8,40},0,0,0},
{"embed_tokens.weight",2,{8,40},0,0,0},
{"layers.0.input_layernorm.weight",1,{8},13,0,0},
{"layers.0.linear_attn.A_log",1,{4},2,0,0},
{"layers.0.linear_attn.conv1d.weight",3,{4,1,32},3,0,1},
{"layers.0.linear_attn.dt_bias",1,{4},1,0,0},
{"layers.0.linear_attn.in_proj_a.weight",2,{8,4},9,0,0},
{"layers.0.linear_attn.in_proj_b.weight",2,{8,4},8,0,0},
{"layers.0.linear_attn.in_proj_qkv.weight",2,{8,32},6,0,0},
{"layers.0.linear_attn.in_proj_z.weight",2,{8,16},7,0,0},
{"layers.0.linear_attn.norm.weight",1,{4},4,1,0},
{"layers.0.linear_attn.out_proj.weight",2,{16,8},5,0,0},
{"layers.0.mlp.down_proj.weight",2,{12,8},12,0,0},
{"layers.0.mlp.gate_proj.weight",2,{8,12},10,0,0},
{"layers.0.mlp.up_proj.weight",2,{8,12},11,0,0},
{"layers.0.post_attention_layernorm.weight",1,{8},14,0,0},
{"layers.1.input_layernorm.weight",1,{8},24,0,0},
{"layers.1.mlp.down_proj.weight",2,{12,8},23,0,0},
{"layers.1.mlp.gate_proj.weight",2,{8,12},21,0,0},
{"layers.1.mlp.up_proj.weight",2,{8,12},22,0,0},
{"layers.1.post_attention_layernorm.weight",1,{8},25,0,0},
{"layers.1.self_attn.k_norm.weight",1,{8},20,0,0},
{"layers.1.self_attn.k_proj.weight",2,{8,16},16,0,0},
{"layers.1.self_attn.o_proj.weight",2,{32,8},18,0,0},
{"layers.1.self_attn.q_norm.weight",1,{8},19,0,0},
{"layers.1.self_attn.q_proj.weight",2,{8,64},15,0,0},
{"layers.1.self_attn.v_proj.weight",2,{8,16},17,0,0},
{"layers.2.input_layernorm.weight",1,{8},38,0,0},
{"layers.2.linear_attn.A_log",1,{4},27,0,0},
{"layers.2.linear_attn.conv1d.weight",3,{4,1,32},28,0,0},
{"layers.2.linear_attn.dt_bias",1,{4},26,0,0},
{"layers.2.linear_attn.in_proj_a.weight",2,{8,4},34,0,0},
{"layers.2.linear_attn.in_proj_b.weight",2,{8,4},33,0,0},
{"layers.2.linear_attn.in_proj_qkv.weight",2,{8,32},31,0,0},
{"layers.2.linear_attn.in_proj_z.weight",2,{8,16},32,0,0},
{"layers.2.linear_attn.norm.weight",1,{4},29,1,0},
{"layers.2.linear_attn.out_proj.weight",2,{16,8},30,0,0},
{"layers.2.mlp.down_proj.weight",2,{12,8},37,0,0},
{"layers.2.mlp.gate_proj.weight",2,{8,12},35,0,0},
{"layers.2.mlp.up_proj.weight",2,{8,12},36,0,0},
{"layers.2.post_attention_layernorm.weight",1,{8},39,0,0},
{"layers.3.input_layernorm.weight",1,{8},49,0,0},
{"layers.3.mlp.down_proj.weight",2,{12,8},48,0,0},
{"layers.3.mlp.gate_proj.weight",2,{8,12},46,0,0},
{"layers.3.mlp.up_proj.weight",2,{8,12},47,0,0},
{"layers.3.post_attention_layernorm.weight",1,{8},50,0,0},
{"layers.3.self_attn.k_norm.weight",1,{8},45,0,0},
{"layers.3.self_attn.k_proj.weight",2,{8,16},41,0,0},
{"layers.3.self_attn.o_proj.weight",2,{32,8},43,0,0},
{"layers.3.self_attn.q_norm.weight",1,{8},44,0,0},
{"layers.3.self_attn.q_proj.weight",2,{8,64},40,0,0},
{"layers.3.self_attn.v_proj.weight",2,{8,16},42,0,0},
{"norm.weight",1,{8},51,0,0},
};
static int visual_text_fixture(const char *path,int mode){
 gguf_writer *w=gguf_write_open(path);if(!w)return -1;
 gguf_write_kv_str(w,"general.architecture","qwen35");
 gguf_write_kv_str(w,"general.name","Qwen3.5 dense text, source tensor layout");
 gguf_write_kv_u32(w,"qwen35.source_layout_version",1);
 gguf_write_kv_u32(w,"qwen35.block_count",4);
 gguf_write_kv_u32(w,"qwen35.embedding_length",8);
 gguf_write_kv_u32(w,"qwen35.feed_forward_length",12);
 gguf_write_kv_u32(w,"qwen35.vocab_size",40);
 gguf_write_kv_u32(w,"qwen35.context_length",128);
 gguf_write_kv_u32(w,"qwen35.attention.head_count",4);
 gguf_write_kv_u32(w,"qwen35.attention.head_count_kv",2);
 gguf_write_kv_u32(w,"qwen35.attention.head_dim",8);
 gguf_write_kv_u32(w,"qwen35.rope.dimension_count",8);
 gguf_write_kv_u32(w,"qwen35.linear.key_heads",2);
 gguf_write_kv_u32(w,"qwen35.linear.value_heads",4);
 gguf_write_kv_u32(w,"qwen35.linear.key_dim",4);
 gguf_write_kv_u32(w,"qwen35.linear.value_dim",4);
 gguf_write_kv_u32(w,"qwen35.linear.conv_kernel",4);
 gguf_write_kv_u32(w,"qwen35.layer.0.full_attention",0);
 gguf_write_kv_u32(w,"qwen35.layer.1.full_attention",1);
 gguf_write_kv_u32(w,"qwen35.layer.2.full_attention",0);
 gguf_write_kv_u32(w,"qwen35.layer.3.full_attention",1);
 gguf_write_kv_f32(w,"qwen35.rms_epsilon",1e-06);
 gguf_write_kv_f32(w,"qwen35.rope.freq_base",10000.0);
 int count=(int)(sizeof(visual_text_tensors)/sizeof(visual_text_tensors[0]));
 for(int t=0;t<count;t++)if(mode!=1 || !visual_text_tensors[t].drop)
  gguf_write_tensor_decl(w,visual_text_tensors[t].name,visual_text_tensors[t].ndim,visual_text_tensors[t].dims,GGUF_TYPE_F32);
 for(int t=0;t<count;t++)if(mode!=1 || !visual_text_tensors[t].drop){
  const visual_text_tensor *g=&visual_text_tensors[t];uint64_t n=1;for(uint32_t d=0;d<g->ndim;d++)n*=g->dims[d];
  float *v=malloc(n*sizeof(float));if(!v){gguf_write_abort(w);return -1;}
  for(uint64_t i=0;i<n;i++){v[i]=(float)((int)((i*7+(uint64_t)g->param*3)%23)-11)*0.0175f;if(g->plus_one)v[i]+=1.0f;}
  int rc=gguf_write_tensor_f32(w,g->name,v,n);free(v);if(rc){gguf_write_abort(w);return -1;}
 }
 return gguf_write_close(w);
}
static const float visual_text_golden_residual[4][56]={
{-0x1.3435100000000p-3f,0x1.21f9b80000000p-6f,-0x1.622c1c0000000p-5f,-0x1.ea92040000000p-5f,-0x1.c05ac40000000p-3f,0x1.0e785c0000000p-4f,0x1.d5f8380000000p-3f,0x1.11a1dc0000000p-6f,-0x1.e328da0000000p-2f,-0x1.cf87c80000000p-2f,-0x1.03eab40000000p-1f,-0x1.764f380000000p-2f,-0x1.6df7380000000p-2f,-0x1.51c7320000000p-2f,-0x1.36a9560000000p-2f,-0x1.5a395c0000000p-2f,-0x1.c333e60000000p-3f,-0x1.c5ed5a0000000p-3f,-0x1.51a28e0000000p-2f,-0x1.5bee000000000p-3f,-0x1.236ed20000000p-3f,-0x1.2ceed80000000p-5f,0x1.51e5dc0000000p-7f,-0x1.1e05720000000p-5f,0x1.c1a5e80000000p-7f,0x1.36ac2e0000000p-7f,0x1.4937620000000p-5f,0x1.1eeff40000000p-4f,0x1.1de02c0000000p-4f,0x1.d671760000000p-3f,0x1.e9727e0000000p-3f,0x1.b605be0000000p-3f,0x1.8ee6ac0000000p-3f,0x1.02f2be0000000p-2f,0x1.37a9660000000p-2f,0x1.63cd3a0000000p-2f,0x1.6f41c20000000p-2f,0x1.f71d740000000p-2f,0x1.ef86ca0000000p-2f,0x1.9054ae0000000p-2f,-0x1.23f9240000000p-5f,0x1.3bcbb60000000p-3f,-0x1.401cf80000000p-3f,-0x1.c27f500000000p-5f,0x1.5530720000000p-5f,-0x1.0bc1400000000p-6f,0x1.1e0d940000000p-3f,0x1.a470f40000000p-5f,-0x1.0e2b1a0000000p-3f,-0x1.116b160000000p-3f,0x1.4dd04c0000000p-4f,0x1.4143860000000p-3f,-0x1.7dd8c60000000p-3f,0x1.6d200a0000000p-3f,0x1.54e1600000000p-4f,-0x1.9ce3ec0000000p-3f},
{-0x1.a99b300000000p-4f,0x1.5b02740000000p-4f,-0x1.25ac400000000p-3f,-0x1.72779e0000000p-6f,-0x1.23174e0000000p-3f,0x1.004e6c0000000p-4f,0x1.c576f20000000p-3f,-0x1.dc19f20000000p-5f,-0x1.c1b6f40000000p-2f,-0x1.bcfa8a0000000p-2f,-0x1.fcc0200000000p-2f,-0x1.790c540000000p-2f,-0x1.48e0860000000p-2f,-0x1.607c220000000p-2f,-0x1.1ad7c80000000p-2f,-0x1.9dafb60000000p-2f,-0x1.92f80a0000000p-3f,-0x1.bfc0080000000p-3f,-0x1.445d140000000p-2f,-0x1.5fc8e40000000p-3f,-0x1.b1c1480000000p-4f,-0x1.57c6d80000000p-5f,0x1.c302660000000p-6f,-0x1.74bca00000000p-4f,0x1.2fc2bc0000000p-6f,0x1.3390fa0000000p-5f,0x1.186a880000000p-5f,0x1.1ece9a0000000p-4f,0x1.6644200000000p-4f,0x1.92b6d20000000p-3f,0x1.1baf5a0000000p-2f,0x1.5809400000000p-3f,0x1.88796c0000000p-3f,0x1.2dcb2a0000000p-2f,0x1.25d89e0000000p-2f,0x1.6141280000000p-2f,0x1.6aa3180000000p-2f,0x1.e774460000000p-2f,0x1.fe393c0000000p-2f,0x1.6407ea0000000p-2f,-0x1.65d86c0000000p-4f,0x1.9896e40000000p-3f,-0x1.3eb2c80000000p-3f,-0x1.b9b5ac0000000p-4f,0x1.0025600000000p-3f,0x1.b917a60000000p-5f,0x1.1b6e9e0000000p-4f,0x1.b70d480000000p-5f,-0x1.1f66820000000p-4f,-0x1.7f9fe00000000p-3f,0x1.1120260000000p-3f,0x1.0078d80000000p-4f,-0x1.c0265c0000000p-4f,0x1.978dcc0000000p-3f,0x1.2eb1f60000000p-4f,-0x1.2f75800000000p-2f},
{-0x1.89d4e60000000p-3f,0x1.0e66480000000p-5f,-0x1.3142d20000000p-3f,-0x1.1d39e00000000p-6f,-0x1.6fd8a40000000p-4f,0x1.5bf5aa0000000p-4f,0x1.c7128c0000000p-3f,-0x1.870e500000000p-5f,-0x1.befae00000000p-2f,-0x1.d8ded20000000p-2f,-0x1.05521e0000000p-1f,-0x1.b64e3e0000000p-2f,-0x1.70667a0000000p-2f,-0x1.7335500000000p-2f,-0x1.c4d3820000000p-3f,-0x1.6003c20000000p-2f,-0x1.6a94e80000000p-3f,-0x1.b6664c0000000p-3f,-0x1.3f012a0000000p-2f,-0x1.031f0c0000000p-2f,-0x1.5e16b80000000p-3f,-0x1.b8c7380000000p-5f,0x1.4747ec0000000p-4f,-0x1.de09a40000000p-5f,-0x1.f182100000000p-5f,0x1.a102960000000p-5f,0x1.7b020a0000000p-4f,0x1.6a3e080000000p-4f,0x1.c65a7c0000000p-4f,0x1.a00b860000000p-3f,0x1.006cbe0000000p-2f,0x1.55947e0000000p-3f,0x1.0e0fd20000000p-3f,0x1.16c7ae0000000p-2f,0x1.ba4dee0000000p-3f,0x1.952ae20000000p-2f,0x1.898c6c0000000p-2f,0x1.fd68d00000000p-2f,0x1.1636420000000p-1f,0x1.9b78440000000p-2f,-0x1.69274a0000000p-4f,0x1.83bf200000000p-3f,-0x1.411efc0000000p-4f,-0x1.0559580000000p-3f,0x1.b2cce60000000p-4f,-0x1.38ea740000000p-6f,0x1.bd3dbc0000000p-7f,-0x1.8f25840000000p-6f,-0x1.42c75e0000000p-4f,-0x1.69c9a60000000p-2f,0x1.80ad240000000p-3f,0x1.4884a00000000p-2f,0x1.83a5800000000p-5f,0x1.d4649a0000000p-3f,-0x1.60e4740000000p-5f,-0x1.e73f100000000p-2f},
{-0x1.f8e86a0000000p-3f,0x1.172a480000000p-4f,-0x1.7146a00000000p-5f,-0x1.4a53380000000p-3f,0x1.6588e80000000p-6f,0x1.0a74c80000000p-4f,0x1.b9b97e0000000p-3f,-0x1.18a2b00000000p-4f,-0x1.b009980000000p-2f,-0x1.fa19a80000000p-2f,-0x1.d2942a0000000p-2f,-0x1.007d340000000p-1f,-0x1.1a75900000000p-2f,-0x1.b082500000000p-2f,-0x1.89ef840000000p-3f,-0x1.91d4760000000p-2f,-0x1.55780c0000000p-3f,-0x1.e4e1080000000p-3f,-0x1.20ad900000000p-2f,-0x1.4c27940000000p-2f,-0x1.4a87880000000p-4f,-0x1.8c5a2c0000000p-4f,0x1.ebbd5e0000000p-4f,-0x1.a9ccbc0000000p-4f,-0x1.06906c0000000p-4f,0x1.31da480000000p-5f,0x1.17d3300000000p-3f,0x1.d3b8e40000000p-8f,0x1.6545800000000p-3f,0x1.7ca5600000000p-3f,0x1.3ba6d60000000p-2f,0x1.b95ba80000000p-4f,0x1.0819820000000p-3f,0x1.07b5480000000p-2f,0x1.0a4c640000000p-2f,0x1.489efa0000000p-2f,0x1.bff5740000000p-2f,0x1.debf500000000p-2f,0x1.3797780000000p-1f,0x1.6450100000000p-2f,-0x1.ae97040000000p-4f,0x1.79ae7c0000000p-3f,-0x1.5c471e0000000p-5f,-0x1.efb8dc0000000p-5f,0x1.1c7eb20000000p-5f,0x1.3958280000000p-4f,0x1.3b0e800000000p-9f,-0x1.2b15380000000p-4f,-0x1.1d0baa0000000p-4f,-0x1.885bb00000000p-2f,0x1.aba8640000000p-3f,0x1.1682180000000p-2f,0x1.8263800000000p-4f,0x1.8add060000000p-3f,0x1.c468560000000p-5f,-0x1.14ed000000000p-1f},
};
static const float visual_text_golden_logits[40]={0x1.5fadd60000000p-1f,-0x1.2aec340000000p-2f,0x1.4886d60000000p-1f,0x1.9043d00000000p-5f,-0x1.4dd0d80000000p-6f,0x1.cb28800000000p-4f,-0x1.3ecd840000000p-1f,0x1.54e2e80000000p-1f,-0x1.1e6cb00000000p-1f,0x1.3dbbe60000000p-1f,0x1.c729900000000p-6f,-0x1.9a9d340000000p-3f,0x1.74d0fa0000000p-4f,-0x1.4998740000000p-1f,0x1.e0a8b40000000p-3f,-0x1.2937a00000000p-1f,0x1.32f0f60000000p-1f,-0x1.1556520000000p-2f,-0x1.c5c8f60000000p-3f,0x1.1e79740000000p-4f,0x1.71a3800000000p-11f,0x1.b57cf00000000p-3f,-0x1.3402900000000p-1f,0x1.5fadd60000000p-1f,-0x1.2aec340000000p-2f,0x1.4886d60000000p-1f,0x1.9043d00000000p-5f,-0x1.4dd0d80000000p-6f,0x1.cb28800000000p-4f,-0x1.3ecd840000000p-1f,0x1.54e2e80000000p-1f,-0x1.1e6cb00000000p-1f,0x1.3dbbe60000000p-1f,0x1.c729900000000p-6f,-0x1.9a9d340000000p-3f,0x1.74d0fa0000000p-4f,-0x1.4998740000000p-1f,0x1.e0a8b40000000p-3f,-0x1.2937a00000000p-1f,0x1.32f0f60000000p-1f};
