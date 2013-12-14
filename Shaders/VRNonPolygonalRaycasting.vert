varying vec3 v;

void main(void)
{

	gl_TexCoord[0] = gl_TextureMatrix[7] * gl_MultiTexCoord0;
	//transform vertex position into homogeneous clip space
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	v = vec3(gl_ModelViewMatrix * gl_Vertex);

}