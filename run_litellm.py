#!/usr/bin/env python3
# LiteLLM Config Generator - Developed by acidvegas (https://github.com/acidvegas)

import json
import logging
import os
import subprocess
import urllib.request

try:
	import apv
except ImportError:
	raise SystemExit('missing apv library (pip install apv)')

try:
	from dotenv import load_dotenv
	load_dotenv()
except ImportError:
	raise SystemExit('missing dotenv library (pip install python-dotenv)')


# Create API key variables
class API_KEY:
	anthropic  = os.getenv('ANTHROPIC_API_KEY')
	openai     = os.getenv('OPENAI_API_KEY')
	gemini     = os.getenv('GEMINI_API_KEY')
	xai        = os.getenv('XAI_API_KEY')
	perplexity = os.getenv('PERPLEXITYAI_API_KEY')
	elevenlabs = os.getenv('ELEVEN_LABS_API_KEY')


def make_request(url: str, headers: dict = {}) -> dict:
	'''
	Make a request to the given URL with the given headers and return the JSON response

	:param url: The URL to make the request to
	:param headers: The headers to include in the request
	'''

	# Create the request
	req = urllib.request.Request(url, headers=headers)

	# Make the request
	with urllib.request.urlopen(req) as resp:
		return json.load(resp)
	

class ModelFetcher:

	def __init__(self):
		self.config = ['model_list:']
		self.counts = {}


	def add_model(self, model_name: str, litellm_params: dict):
		'''
		Add a model to the config

		:param model_name: The name of the model
		:param litellm_params: The LiteLLM model parameters to pass to the model
		'''

		# Add the model_name to the config
		self.config.append(f'  - model_name: {model_name}')
		self.config.append(f'    litellm_params:')
		
		# Add the litellm_params to the config
		for key, value in litellm_params.items():
			self.config.append(f'      {key}: {value}')


	def fetch_all_models(self):
		'''Fetch all models from all providers'''

		# Fetch all the models
		for function in self.supported_providers(functions=True):

			# Get the provider name
			provider = function.split('_')[2]

			# Check if we have an API key for this provider
			if not getattr(API_KEY, provider, None):
				# Ollama doesn't need an API key, just the base URL
				if provider == 'ollama' and os.getenv('OLLAMA_API_BASE'):
					pass
				else:
					logging.warning(f'Skipping {provider}: API key not set')
					continue

			# Run the function
			try:
				logging.info(f'Running {function}...')
				getattr(self, function)()
			except Exception as e:
				logging.error(f'Error fetching {function}: {e}')


	def fetch_models_anthropic(self):
		'''Add a list of models from Anthropic to the LiteLLM config'''

		# Make the request to the Anthropic API
		data = make_request('https://api.anthropic.com/v1/models', headers={'x-api-key': API_KEY.anthropic, 'anthropic-version': '2023-06-01'})

		# Parse the models list
		models = data.get('data', [])

		# Update the model count for this provider
		self.counts['Anthropic'] = len(models)

		logging.info(f'Found {len(models):,} Anthropic models')

		# Add the models to the config
		for model in models:
			self.add_model(model['id'], {'model': f'anthropic/{model["id"]}'})


	def fetch_models_elevenlabs(self):
		'''Add a list of models from ElevenLabs to the LiteLLM config'''

		# Make the request to the ElevenLabs API
		data = make_request('https://api.elevenlabs.io/v1/models', headers={'xi-api-key': API_KEY.elevenlabs})

		# Parse the models list
		models = [m for m in data if m.get('can_do_text_to_speech')]

		# Update the model count for this provider
		self.counts['ElevenLabs'] = len(models)

		logging.info(f'Found {len(models):,} ElevenLabs TTS models')

		# Add the models to the config
		for model in models:
			self.add_model(model['model_id'], {'model': f'elevenlabs/{model["model_id"]}'})


	def fetch_models_gemini(self, next_page_token: str = None):
		'''
		Add a list of models from Gemini to the LiteLLM config

		:param next_page_token: The next page token to use for pagination
		'''

		# Build the URL
		url = f'https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY.gemini}'

		# Add the next page token to the URL if it is provided
		if next_page_token:
			url += f'&pageToken={next_page_token}'

		# Make the request to the Gemini API
		data = make_request(url)

		# Parse the models list
		models = data.get('models', [])

		# Update the model count for this provider
		self.counts['Gemini'] = len(models)

		logging.info(f'Found {len(models):,} Gemini models')

		# Add the models to the config
		for model in models:
			self.add_model(model['name'], {'model': f'gemini/{model["name"]}'})

		# Get the next page token (if there is one) and fetch the next page of models
		if (next_page_token := data.get('nextPageToken')):
			self.fetch_models_gemini(next_page_token)


	def fetch_models_perplexity(self):
		'''Add a list of models from Perplexity to the LiteLLM config'''

		models = ['sonar', 'sonar-pro', 'sonar-reasoning', 'sonar-reasoning-pro', 'sonar-deep-research']

		# Update the model count for this provider
		self.counts['Perplexity'] = len(models)

		logging.info(f'Found {len(models):,} Perplexity models')

		# Add the models to the config
		for model in models:
			self.add_model(model, {'model': f'perplexity/{model}'})


	def fetch_models_ollama(self):
		'''Add a list of models from Ollama to the LiteLLM config'''

		ollama_endpoint = os.getenv('OLLAMA_API_BASE') or 'http://localhost:11434'

		# Make the request to the Ollama API
		data = make_request(f'{ollama_endpoint}/api/tags')

		# Parse the models list
		models = data.get('models', [])

		# Update the model count for this provider
		self.counts['Ollama'] = len(models)

		logging.info(f'Found {len(models):,} Ollama models')

		# Add the models to the config
		for model in models:
			self.add_model(model['name'], {'model': f'ollama/{model["name"]}', 'api_base': ollama_endpoint})


	def fetch_models_openai(self):
		'''Add a list of models from OpenAI to the LiteLLM config'''

		# Make the request to the OpenAI API
		data = make_request('https://api.openai.com/v1/models', headers={'Authorization': f'Bearer {API_KEY.openai}'})

		# Parse the models list
		models = data.get('data', [])

		# Update the model count for this provider
		self.counts['OpenAI'] = len(models)

		logging.info(f'Found {len(models):,} OpenAI models')

		# Add the models to the config
		for model in models:
			self.add_model(model['id'], {'model': f'openai/{model["id"]}'})


	def fetch_models_xai(self):
		'''Add a list of models from xAI to the LiteLLM config'''

		# Make the request to the xAI API
		data = make_request('https://api.x.ai/v1/models', headers={'Authorization': f'Bearer {API_KEY.xai}', 'User-Agent': 'curl/8.0'})

		# Parse the models list
		models = data.get('data', [])

		# Update the model count for this provider
		self.counts['xAI'] = len(models)

		logging.info(f'Found {len(models):,} xAI models')

		# Add the models to the config
		for model in models:
			self.add_model(model['id'], {'model': f'xai/{model["id"]}'})


	def supported_providers(self, functions: bool = False) -> list[str]:
		'''Return a list of supported providers'''

		return [f for f in dir(self) if f.startswith('fetch_models_')] if functions else [f.split('_')[2] for f in dir(self) if f.startswith('fetch_models_')]



if __name__ == '__main__':
	import argparse

	# Parse the command line arguments
	parser = argparse.ArgumentParser(description='LiteLLM Config Generator')
	parser.add_argument('-d', '--debug', action='store_true', help='Show debug information')
	parser.add_argument('-l', '--list', action='store_true', help='List the supported providers')
	parser.add_argument('-p', '--provider', help='Fetch models for a specific provider', nargs='+')
	args = parser.parse_args()

	# Setup logging
	apv.setup_logging(level='DEBUG' if args.debug else 'INFO', show_details=True)
	
	# Create the ModelFetcher instance
	model_fetcher = ModelFetcher()

	if args.list:
		# List the supported providers
		logging.info('Supported providers:')
		for provider in model_fetcher.supported_providers():
			logging.info(f'  {provider}')
		exit()

	if not args.provider:
		# Fetch all the models
		model_fetcher.fetch_all_models()
	else:
		for item in args.provider:
			if item not in model_fetcher.supported_providers():
				raise ValueError(f'Provider {item} not supported')
			# Run the function to fetch the models for the specified provider
			getattr(model_fetcher, f'fetch_models_{item}')()

	# Write the config to file
	with open('litellm_config.yaml', 'w') as f:
		f.write('\n'.join(model_fetcher.config))

	# Print the summary
	logging.info('Summary:')
	for provider, count in model_fetcher.counts.items():
		logging.info(f'  {provider.ljust(12)} {str(count).rjust(5)}')
	logging.info(f'  {"Total".ljust(12)} {str(sum(model_fetcher.counts.values())).rjust(5)}')
	logging.info('Config written to litellm_config.yaml')

	# Get the path to the config file
	config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'litellm_config.yaml')

	# Check if the config file exists
	if not os.path.exists(config_path):
		logging.error('config file not found')
		exit(1)

	logging.info('Starting LiteLLM on port 4000...')
	logging.warning('Press Ctrl+C to stop')

	# Start LiteLLM
	try:
		subprocess.run(['litellm', '--config', config_path, '--port', '4000'])
	except KeyboardInterrupt:
		logging.warning('LiteLLM stopped')