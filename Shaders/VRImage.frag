uniform sampler2D image;

void main()
{
	vec4 color = texture2D(image, vec2(gl_TexCoord[0].s, gl_TexCoord[0].t));
	gl_FragColor = color;
}