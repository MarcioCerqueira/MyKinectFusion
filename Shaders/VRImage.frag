uniform sampler2D image;

void main()
{
	//vec4 color = texture2D(image, vec2(gl_TexCoord[0].s, gl_TexCoord[0].t));
	float depth  = texture2D(image, vec2(gl_TexCoord[0].s, 1 - gl_TexCoord[0].t)).r;
	depth = 1.0 - depth;
	gl_FragColor = vec4(depth, depth, depth, 1);
}