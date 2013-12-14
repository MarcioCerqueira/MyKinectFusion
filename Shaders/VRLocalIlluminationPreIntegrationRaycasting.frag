uniform sampler3D volume;
uniform sampler3D minMaxOctree;
uniform sampler2D transferFunction;
uniform sampler2D noise;
uniform sampler2D frontFrameBuffer;
uniform sampler2D backFrameBuffer;
uniform float stepSize;
uniform float earlyRayTerminationThreshold;
uniform vec3 camera;
varying vec3 v;
uniform int MIP;
uniform int forwardDifference;
uniform int stochasticJithering;
uniform int windowWidth;
uniform int windowHeight;

//Blinn-Phong Illumination
vec3 BlinnPhongShading(vec3 L, vec3 N, vec3 V) 
{

	vec3 H = normalize(L + V);
	
	vec4 ambient = gl_FrontLightProduct[0].ambient;
	
	float diffuseLight = max(dot(L, N), 0);
	vec4 diffuse = gl_FrontLightProduct[0].diffuse * diffuseLight;
	
	float specularLight = pow(max(dot(H, N), 0), 60);
	if(diffuseLight <= 0) specularLight = 0;
	vec4 specular = gl_FrontLightProduct[0].specular * specularLight;
	
	return gl_FrontLightModelProduct.sceneColor + ambient + diffuse + specular;
}

vec4 computeIllumination(vec4 scalar, vec3 position) 
{

	if(scalar.a > 0.075) {
		float delta = 0.01;
		vec3 sample1, sample2;
		vec3 alpha1, alpha2;
		sample2.x = texture3D(volume, vec3(position + vec3(delta, 0, 0))).x;
		sample2.y = texture3D(volume, vec3(position + vec3(0, delta, 0))).x;
		sample2.z = texture3D(volume, vec3(position + vec3(0, 0, delta))).x;
		alpha2.x = texture2D(transferFunction, vec2(sample2.x, sample2.x)).a;
		alpha2.y = texture2D(transferFunction, vec2(sample2.y, sample2.y)).a;
		alpha2.z = texture2D(transferFunction, vec2(sample2.z, sample2.z)).a;
		
		if(forwardDifference == 0) {
			sample1.x = texture3D(volume, vec3(position - vec3(delta, 0, 0))).x;
			sample1.y = texture3D(volume, vec3(position - vec3(0, delta, 0))).x;
			sample1.z = texture3D(volume, vec3(position - vec3(0, 0, delta))).x;
			alpha1.x = texture2D(transferFunction, vec2(sample1.x, sample1.x)).a;
			alpha1.y = texture2D(transferFunction, vec2(sample1.y, sample1.y)).a;
			alpha1.z = texture2D(transferFunction, vec2(sample1.z, sample1.z)).a;
		}

		//central difference and normalization
		//for(int i = 0; i < 1; i++) {
			vec3 N;
			if(forwardDifference == 1)
				N = normalize(scalar - alpha2);
			else
				N = normalize(alpha2 - alpha1);
			vec3 L = normalize(gl_LightSource[0].position.xyz - v); 
			vec3 V = normalize(-v);
			scalar.rgb += BlinnPhongShading(L, N, V).rgb;
		//}
	}

	return scalar;
}

void main (void)  
{

	vec4 value;
	vec2 scalar = vec2(0, 0);
	vec4 src = vec4(0, 0, 0, 0);
	vec4 rayStart = texture2D(frontFrameBuffer, vec2(gl_FragCoord.x/windowWidth, gl_FragCoord.y/windowHeight));
	vec4 rayEnd = texture2D(backFrameBuffer, vec2(gl_FragCoord.x/windowWidth, gl_FragCoord.y/windowHeight));
	if(rayStart == rayEnd)
		discard;
	//Initialize accumulated color and opacity
	vec4 dst = vec4(0, 0, 0, 0);
	//Determine volume entry position
	vec3 position = vec3(rayStart);
	vec3 direction = vec3(rayEnd) - vec3(rayStart);
	float len = length(direction); // the length from front to back is calculated and used to terminate the ray
    direction = normalize(direction);
	if(stochasticJithering == 1)
		position = position + direction * texture2D(noise, gl_FragCoord.xy / 256).x/64;
	float dirLength = normalize(direction);
	//Loop for ray traversal
	float maxStepSize = 0.04f;
	//float maxStepSize = 4 * stepSize;
	float accLength = 0;
	vec4 maxOpacity;
	
	for(int i = 0; i < 100; i++) //Some large number
	{
		
		maxOpacity = texture3D(minMaxOctree, position);
		if(maxOpacity.g > 0) {

			//Data access to scalar value in 3D volume texture
			value = texture3D(volume, position);
			
			vec3 s = -stepSize * 0.5;
			position = position + direction * s;
			value = texture3D(volume, position);
			if(value.a > 0.1) s *= 0.5;
			else	s *= -0.5;
			position = position + direction * s;
			value = texture3D(volume, position);

			scalar.y = value.a;
			src = texture2D(transferFunction, scalar.xy);
			src = computeIllumination(src, position);
			
			//Front-to-back compositing
			if(MIP == 0) {
				if(src.a > 0.1)
					dst = (1 - dst.a) * src + dst;
			} else
				dst = max(dst, src);
			//Advance ray position along ray direction
			position = position + direction * stepSize;
			accLength += dirLength * stepSize;
			
			//Save previous scalar value
			scalar.x = src.a;

		} else {
		
			position = position + direction * maxStepSize;
			accLength += dirLength * maxStepSize;
		
		}
		
		//Additional termination condition for early ray termination
		if(dst.a > earlyRayTerminationThreshold)
			break;
		
		//Ray termination: Test if outside volume...
		if(accLength > len)
			break;
		
	}

	gl_FragColor = dst;
}