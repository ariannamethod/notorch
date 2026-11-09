/* Frozen independent Transformers 5.8.0 F32/eager reference, written by tests/qwen35_fixture_gen.py gqa.
 * E8 F12 H4 KV2 HD4 RoPE2; K-head2 V-head4 KD4 VD4 conv4; delta,full,delta,full.
 * Parameters: ((arange(n)*7 + parameter_index*3)%23-11)*0.0175f;
 * linear_attn.norm adds 1; the writer computes them, only the outputs are listed. Tokens: 0,3,5,12,2,6,7. */
typedef struct {const char *name;uint32_t ndim;uint64_t dims[3];int param,plus_one,drop;} gqa_tensor;
static const gqa_tensor gqa_tensors[]={
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
{"layers.1.self_attn.k_norm.weight",1,{4},20,0,0},
{"layers.1.self_attn.k_proj.weight",2,{8,8},16,0,0},
{"layers.1.self_attn.o_proj.weight",2,{16,8},18,0,0},
{"layers.1.self_attn.q_norm.weight",1,{4},19,0,0},
{"layers.1.self_attn.q_proj.weight",2,{8,32},15,0,0},
{"layers.1.self_attn.v_proj.weight",2,{8,8},17,0,0},
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
{"layers.3.self_attn.k_norm.weight",1,{4},45,0,0},
{"layers.3.self_attn.k_proj.weight",2,{8,8},41,0,0},
{"layers.3.self_attn.o_proj.weight",2,{16,8},43,0,0},
{"layers.3.self_attn.q_norm.weight",1,{4},44,0,0},
{"layers.3.self_attn.q_proj.weight",2,{8,32},40,0,0},
{"layers.3.self_attn.v_proj.weight",2,{8,8},42,0,0},
{"norm.weight",1,{8},51,0,0},
};
static int gqa_fixture(const char *path,int mode){
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
 gguf_write_kv_u32(w,"qwen35.attention.head_dim",4);
 gguf_write_kv_u32(w,"qwen35.rope.dimension_count",2);
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
 int count=(int)(sizeof(gqa_tensors)/sizeof(gqa_tensors[0]));
 for(int t=0;t<count;t++)if(mode!=1 || !gqa_tensors[t].drop)
  gguf_write_tensor_decl(w,gqa_tensors[t].name,gqa_tensors[t].ndim,gqa_tensors[t].dims,GGUF_TYPE_F32);
 for(int t=0;t<count;t++)if(mode!=1 || !gqa_tensors[t].drop){
  const gqa_tensor *g=&gqa_tensors[t];uint64_t n=1;for(uint32_t d=0;d<g->ndim;d++)n*=g->dims[d];
  float *v=malloc(n*sizeof(float));if(!v){gguf_write_abort(w);return -1;}
  for(uint64_t i=0;i<n;i++){v[i]=(float)((int)((i*7+(uint64_t)g->param*3)%23)-11)*0.0175f;if(g->plus_one)v[i]+=1.0f;}
  int rc=gguf_write_tensor_f32(w,g->name,v,n);free(v);if(rc){gguf_write_abort(w);return -1;}
 }
 return gguf_write_close(w);
}
static const float gqa_golden_residual[4][56]={
{-0x1.3435100000000p-3f,0x1.21f9ba0000000p-6f,-0x1.622c140000000p-5f,-0x1.ea92040000000p-5f,-0x1.c05ac40000000p-3f,0x1.0e785e0000000p-4f,0x1.d5f8380000000p-3f,0x1.11a1cc0000000p-6f,-0x1.4a535c0000000p-4f,0x1.abee520000000p-3f,0x1.7487380000000p-2f,-0x1.89c19a0000000p-4f,0x1.fff9a80000000p-6f,0x1.5e1ffa0000000p-2f,-0x1.d1ec540000000p-2f,-0x1.4557980000000p-3f,-0x1.7a5aa00000000p-5f,0x1.5305680000000p-7f,0x1.549b200000000p-4f,-0x1.c03db60000000p-3f,-0x1.ad919c0000000p-3f,0x1.aed8800000000p-3f,-0x1.0b42e20000000p-3f,-0x1.e7b4f00000000p-6f,-0x1.592c000000000p-12f,0x1.0affc80000000p-2f,0x1.a02d380000000p-2f,-0x1.0bec680000000p-4f,0x1.8e39060000000p-4f,0x1.76f2720000000p-6f,-0x1.4265280000000p-1f,-0x1.7a17e80000000p-3f,0x1.6631f00000000p-7f,-0x1.16a03a0000000p-2f,-0x1.aec00c0000000p-4f,0x1.302e800000000p-8f,-0x1.a9c2180000000p-3f,0x1.180d740000000p-4f,0x1.ce83e80000000p-2f,-0x1.1b27440000000p-3f,0x1.0697800000000p-11f,0x1.0249fc0000000p-4f,-0x1.3e86680000000p-2f,-0x1.4d370c0000000p-3f,0x1.a48aec0000000p-3f,0x1.99eb6e0000000p-4f,0x1.e954a80000000p-3f,0x1.43abac0000000p-4f,-0x1.33edcc0000000p-2f,-0x1.ecb46e0000000p-3f,0x1.7567f80000000p-5f,0x1.1332fc0000000p-2f,-0x1.0f91f80000000p-4f,0x1.22b25c0000000p-4f,0x1.690d840000000p-3f,0x1.359d500000000p-4f},
{-0x1.ad14880000000p-3f,-0x1.517ebe0000000p-7f,0x1.4344940000000p-3f,-0x1.3396060000000p-4f,-0x1.57d3aa0000000p-3f,0x1.1e1c8a0000000p-3f,0x1.1de14e0000000p-3f,-0x1.6037860000000p-4f,-0x1.9525b60000000p-3f,0x1.92bc980000000p-3f,0x1.0d075e0000000p-1f,-0x1.41a1680000000p-5f,0x1.8e5e5e0000000p-5f,0x1.8c11d80000000p-2f,-0x1.069cd40000000p-1f,-0x1.f7ff080000000p-3f,-0x1.977d580000000p-4f,0x1.2a96dc0000000p-7f,0x1.fc68800000000p-3f,-0x1.be13d00000000p-3f,-0x1.54695c0000000p-3f,0x1.1894240000000p-2f,-0x1.ba385e0000000p-3f,-0x1.1106a00000000p-3f,-0x1.8a057a0000000p-7f,0x1.e660460000000p-3f,0x1.db1e600000000p-2f,0x1.e938000000000p-8f,0x1.5706800000000p-4f,0x1.175c940000000p-4f,-0x1.5a2fee0000000p-1f,-0x1.3765000000000p-2f,0x1.2ecd720000000p-4f,-0x1.a172200000000p-3f,0x1.8d0e5c0000000p-4f,-0x1.2ef9d80000000p-4f,-0x1.2b41460000000p-2f,0x1.4d2e900000000p-4f,0x1.5413880000000p-2f,-0x1.aa09f20000000p-3f,0x1.aeeaa00000000p-7f,0x1.ba3c600000000p-4f,-0x1.c752b20000000p-3f,-0x1.ac47340000000p-4f,0x1.b4b89a0000000p-3f,0x1.fdcf700000000p-4f,0x1.7707740000000p-3f,0x1.e2efd80000000p-8f,-0x1.417b2a0000000p-2f,-0x1.c59a240000000p-3f,0x1.72ed8a0000000p-3f,0x1.fbd8200000000p-3f,-0x1.721b4a0000000p-4f,0x1.9a933e0000000p-4f,0x1.03c8fc0000000p-3f,0x1.8b12040000000p-6f},
{-0x1.1720a80000000p-2f,0x1.1fc7640000000p-5f,0x1.52965c0000000p-5f,-0x1.04a36e0000000p-6f,-0x1.2b38fa0000000p-3f,0x1.09c4380000000p-3f,0x1.50ffd80000000p-3f,-0x1.0555ae0000000p-7f,-0x1.3744500000000p-2f,0x1.54ccc00000000p-9f,0x1.693d9c0000000p-1f,-0x1.321ab80000000p-6f,0x1.1acb400000000p-4f,0x1.f8e04c0000000p-2f,-0x1.6ef8d20000000p-2f,-0x1.eb43840000000p-3f,0x1.c712f60000000p-3f,0x1.b35cdc0000000p-5f,0x1.4242460000000p-3f,-0x1.216d580000000p-2f,-0x1.2e5bc00000000p-2f,0x1.f3c83c0000000p-3f,-0x1.489cbc0000000p-3f,-0x1.69677c0000000p-3f,0x1.9a37200000000p-3f,0x1.3ba1480000000p-3f,0x1.8d94020000000p-2f,-0x1.03638a0000000p-3f,0x1.1d34580000000p-7f,0x1.4095880000000p-3f,-0x1.0d94340000000p-1f,-0x1.3007bc0000000p-2f,-0x1.f3a6340000000p-5f,-0x1.aa031c0000000p-2f,0x1.0471f80000000p-3f,0x1.84cf340000000p-4f,-0x1.e68c360000000p-3f,0x1.83c4900000000p-4f,0x1.08a7e60000000p-2f,-0x1.5dcf840000000p-2f,-0x1.5da4ec0000000p-6f,0x1.74a4ce0000000p-4f,-0x1.6db7400000000p-2f,-0x1.548c2c0000000p-3f,0x1.3140f00000000p-3f,0x1.eb0f9c0000000p-3f,0x1.23288a0000000p-2f,0x1.46ed480000000p-4f,-0x1.6e906a0000000p-2f,-0x1.fb28dc0000000p-3f,0x1.e36b4a0000000p-3f,0x1.3461480000000p-2f,0x1.3ff7de0000000p-7f,0x1.1e55e00000000p-4f,0x1.b245c80000000p-5f,0x1.5f7fc00000000p-4f},
{-0x1.4256560000000p-2f,0x1.5533160000000p-4f,0x1.3e42920000000p-4f,-0x1.8ca4c80000000p-4f,-0x1.b2a98e0000000p-4f,0x1.8a0cf80000000p-6f,0x1.a8001a0000000p-4f,0x1.8c4fa00000000p-4f,-0x1.7a930c0000000p-2f,-0x1.7456a20000000p-5f,0x1.5d4ea40000000p-1f,0x1.aaf6280000000p-7f,0x1.20b6100000000p-3f,0x1.7fbbfc0000000p-2f,-0x1.80d6fa0000000p-2f,-0x1.f937c80000000p-3f,0x1.2412520000000p-3f,0x1.444d900000000p-4f,0x1.216e3c0000000p-3f,-0x1.e728060000000p-3f,-0x1.684f440000000p-3f,0x1.5f37e80000000p-4f,-0x1.a126de0000000p-3f,-0x1.b7d3120000000p-3f,0x1.c231700000000p-4f,0x1.4766000000000p-3f,0x1.66958e0000000p-2f,-0x1.4a96a20000000p-4f,0x1.3fb1200000000p-3f,0x1.2e4dec0000000p-4f,-0x1.15ef8c0000000p-1f,-0x1.7830220000000p-2f,-0x1.66576c0000000p-3f,-0x1.994e6e0000000p-2f,0x1.a6ba4c0000000p-4f,0x1.846e980000000p-3f,-0x1.b562b40000000p-4f,-0x1.3797380000000p-6f,0x1.393c820000000p-2f,-0x1.619d8e0000000p-2f,-0x1.d09f9c0000000p-4f,0x1.f240580000000p-5f,-0x1.5083b00000000p-2f,-0x1.878ca80000000p-4f,0x1.0b41760000000p-2f,0x1.a343400000000p-3f,0x1.06e66e0000000p-2f,-0x1.0870e00000000p-8f,-0x1.7886de0000000p-2f,-0x1.b2be880000000p-3f,0x1.046c760000000p-2f,0x1.0d16760000000p-2f,0x1.3af19c0000000p-6f,-0x1.665b440000000p-6f,-0x1.8945100000000p-9f,0x1.efe0440000000p-4f},
};
static const float gqa_golden_logits[40]={0x1.4ac82a0000000p-1f,-0x1.69c7960000000p-2f,-0x1.0f64b00000000p-3f,0x1.bcff6c0000000p-4f,0x1.766ac80000000p-3f,0x1.c3f9e80000000p-4f,-0x1.4520de0000000p-2f,0x1.4a7dba0000000p-1f,-0x1.4362400000000p-2f,-0x1.108e6c0000000p-3f,0x1.baabfc0000000p-4f,0x1.1ae1d00000000p-3f,0x1.c1a6780000000p-4f,-0x1.45b5bc0000000p-2f,0x1.b041600000000p-4f,-0x1.43f7200000000p-2f,-0x1.11b82c0000000p-3f,-0x1.6932b40000000p-2f,0x1.19b8180000000p-3f,0x1.bf52f00000000p-4f,0x1.77948c0000000p-3f,0x1.adede80000000p-4f,-0x1.448c000000000p-2f,0x1.4ac82a0000000p-1f,-0x1.69c7960000000p-2f,-0x1.0f64b00000000p-3f,0x1.bcff6c0000000p-4f,0x1.766ac80000000p-3f,0x1.c3f9e80000000p-4f,-0x1.4520de0000000p-2f,0x1.4a7dba0000000p-1f,-0x1.4362400000000p-2f,-0x1.108e6c0000000p-3f,0x1.baabfc0000000p-4f,0x1.1ae1d00000000p-3f,0x1.c1a6780000000p-4f,-0x1.45b5bc0000000p-2f,0x1.b041600000000p-4f,-0x1.43f7200000000p-2f,-0x1.11b82c0000000p-3f};
