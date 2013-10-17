uniform sampler3D virtualRGB;

void main (void)  
{

	vec4 virtualRGBColor = textureProj(virtualRGB, gl_TexCoord[0]);
	gl_FragColor = virtualRGBColor;

}