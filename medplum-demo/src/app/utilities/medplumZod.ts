import { z } from "zod";

enum ResourceType {
  Patient = "Patient",
  Observation = "Observation",
}

const humanNameSchema = z.object({
  use: z
    .enum([
      "usual",
      "official",
      "temp",
      "nickname",
      "anonymous",
      "old",
      "maiden",
    ])
    .describe("Identifies the purpose for this name."),
  family: z
    .string()
    .describe("The part of a name that links to the genealogy."),
  given: z.array(z.string()).describe("Given name."),
});

const narrativeSchema = z.object({
  status: z
    .enum(["generated", "extensions", "additional", "empty"])
    .describe(
      "The status of the narrative - whether it's entirely generated (from just the defined data or the extensions too), or whether a human authored it and it may contain additional data."
    ),
  div: z
    .string()
    .describe(
      "The actual narrative content, a stripped down version of XHTML."
    ),
});

const referenceSchema = z.object({
  reference: z
    .string()
    .describe(
      "A reference to a location at which the other resource is found."
    ),
  type: z
    .nativeEnum(ResourceType)
    .describe("The expected type of the target of the reference."),
});

const codingSchema = z.object({
  system: z
    .string()
    .optional()
    .describe(
      "The identification of the code system that defines the meaning of the symbol in the code."
    ),
  version: z
    .string()
    .optional()
    .describe(
      "The version of the code system which was used when choosing this code."
    ),
  code: z
    .string()
    .optional()
    .describe("A symbol in syntax defined by the system."),
  display: z
    .string()
    .optional()
    .describe(
      "A representation of the meaning of the code in the system, following the rules of the system."
    ),
});

const codeableConceptSchema = z.object({
  coding: z
    .array(codingSchema)
    .optional()
    .describe("A reference to a code defined by a terminology system."),
  text: z
    .string()
    .optional()
    .describe(
      "A human language representation of the concept as seen/selected/uttered by the user who entered the data and/or which represents the intended meaning of the user."
    ),
});

const quantitySchema = z.object({
  value: z.number().describe("The value of the measured amount."),
  comparator: z
    .enum(["<", "<=", ">=", ">"])
    .optional()
    .describe(
      "How the value should be understood and represented - whether the actual value is greater or less than the stated value due to measurement issues."
    ),
  unit: z.string().optional().describe("A human-readable form of the unit."),
  system: z
    .string()
    .optional()
    .describe(
      "The identification of the system that provides the coded form of the unit."
    ),
  code: z
    .string()
    .optional()
    .describe(
      "A computer processable form of the unit in some unit representation system."
    ),
});

export const patientCreateSchema = z.object({
  resourceType: z.literal(ResourceType.Patient),
  text: narrativeSchema.describe(
    "A human-readable narrative that contains a summary of the resource and can be used to represent the content of the resource to a human."
  ),
  name: z
    .array(humanNameSchema)
    .describe("A name associated with the individual."),
  gender: z
    .enum(["male", "female", "other", "unknown"])
    .describe(
      "Administrative Gender - the gender that the patient is considered to have for administration and record keeping purposes."
    ),
  birthDate: z.string().describe("The date of birth for the individual."),
});

type PatientCreate = z.infer<typeof patientCreateSchema>;

export const patientSchema;

export const observationCreateSchema = z.object({
  resourceType: z.literal(ResourceType.Observation),
  text: narrativeSchema.describe(
    "A human-readable narrative that contains a summary of the resource and can be used to represent the content of the resource to a human."
  ),
  category: z
    .array(codeableConceptSchema)
    .describe("A code that classifies the general type of observation."),
  code: codeableConceptSchema.describe(
    "Describes what was observed. Sometimes this is called the observation 'name'."
  ),
  subject: referenceSchema.describe("The patient the observation is about."),
  valueQuantity: quantitySchema
    .optional()
    .describe(
      "The information determined as a result of making the observation, if the information has a simple value."
    ),
  valueCodeableConcept: codeableConceptSchema
    .optional()
    .describe(
      "The information determined as a result of making the observation, if the information has a simple value."
    ),
  valueString: z
    .string()
    .optional()
    .describe(
      "The information determined as a result of making the observation, if the information has a simple value."
    ),
  valueBoolean: z
    .boolean()
    .optional()
    .describe(
      "The information determined as a result of making the observation, if the information has a simple value."
    ),
  valueInteger: z
    .number()
    .optional()
    .describe(
      "The information determined as a result of making the observation, if the information has a simple value."
    ),
});
