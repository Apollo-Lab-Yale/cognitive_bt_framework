import xml
import xmlschema


def main():
  print('test')

if __name__ == '__main__' : 
  main()

# Load the XSD schema
schema = xmlschema.XMLSchema('llm_htn_task_decomp/src/bt_validation/bt_schema.xsd')

# Load your XML file
xml_file = 'llm_htn_task_decomp/src/bt_validation/examples/test.xml'

# Validate the XML
try:
    schema.validate(xml_file)
    print("The XML is valid.")
except xmlschema.validators.exceptions.XMLSchemaValidationError as e:
    print("The XML is invalid: ", e)