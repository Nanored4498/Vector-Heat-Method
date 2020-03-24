#version 150

uniform vec3 light_position_eye;
uniform float specular_exponent;
uniform float lighting_factor;

in vec3 position_eye;
in vec3 normal_eye;
in vec4 Kdi;
in vec4 Ksi;

out vec4 outColor;

vec3 Ls = vec3 (0.6, 0.6, 0.6);
vec3 Ld = vec3 (1, 1, 1);

void main() {
	float ni = 30.0; // number of intervals
	float t = round(ni*Kdi.r)/ni; // quantize
	vec3 Kdiq = clamp(vec3(2*(1-t),1-2*t,1-6*t),0,1); // heat map

	vec3 vector_to_light_eye = light_position_eye - position_eye;
	vec3 direction_to_light_eye = normalize(vector_to_light_eye);
	float dot_prod = dot(direction_to_light_eye, normal_eye);
	float clamped_dot_prod = max(dot_prod, 0.0);
	vec3 Id = Ld * Kdiq * clamped_dot_prod;    // Diffuse intensity

	vec3 reflection_eye = reflect(direction_to_light_eye, normal_eye);
	vec3 viewer_to_surface_eye = normalize(position_eye);
	float dot_prod_specular = dot(reflection_eye, viewer_to_surface_eye);
	dot_prod_specular = float(dot_prod >= 0) * max(dot_prod_specular, 0.0);
	float specular_factor = pow(dot_prod_specular, specular_exponent);
	vec3 Is = Ls * vec3(Ksi) * specular_factor;    // specular intensity

	outColor = vec4(lighting_factor * (Is + Id) + (1.0-lighting_factor) * Kdiq, Kdi.a);
}