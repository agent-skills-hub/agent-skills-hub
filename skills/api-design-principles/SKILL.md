---
name: api-design-principles
description: "Designs REST and GraphQL APIs with consistent resource naming, pagination, error schemas, and versioning strategies. Use when designing new endpoints, reviewing OpenAPI/Swagger specs, refactoring API contracts, planning API versioning, or establishing team API standards."
---

# API Design Principles

Designs REST and GraphQL APIs with consistent conventions, clear error handling, and predictable patterns.

## Use this skill when

- Designing new REST or GraphQL endpoints
- Reviewing or writing OpenAPI/Swagger specifications
- Refactoring existing APIs for consistency
- Establishing API standards for a team
- Planning API versioning or migration strategies

## Do not use this skill when

- Implementing framework-specific code (use the framework skill instead)
- Doing infrastructure work without API contract changes

## Instructions

1. Identify consumers, use cases, and constraints (mobile, third-party, internal).
2. Choose API style (REST or GraphQL) and model resources/types.
3. Apply the conventions below for naming, errors, pagination, and versioning.
4. Validate against example requests/responses and review for consistency.

Refer to `resources/implementation-playbook.md` for full patterns and templates.

## REST Naming Conventions

| Pattern | Example | Rule |
|---------|---------|------|
| Collection | `GET /users` | Plural nouns |
| Single resource | `GET /users/{id}` | Identifier in path |
| Sub-resource | `GET /users/{id}/orders` | Nested relationship |
| Action | `POST /users/{id}/activate` | Verb only for non-CRUD |
| Filtering | `GET /users?status=active` | Query params for filters |

## Error Response Schema

```json
{
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "Email address is not valid.",
    "details": [
      { "field": "email", "reason": "must be a valid email address" }
    ]
  }
}
```

Use consistent HTTP status codes: 400 validation, 401 auth, 403 forbidden, 404 not found, 409 conflict, 422 unprocessable, 429 rate limit, 500 server error.

## Pagination

```
GET /users?cursor=eyJpZCI6NDJ9&limit=25

{
  "data": [...],
  "pagination": {
    "next_cursor": "eyJpZCI6Njd9",
    "has_more": true
  }
}
```

Prefer cursor-based over offset-based for large or frequently-changing datasets.

## Resources

- `resources/implementation-playbook.md` — full patterns, checklists, and templates
- `references/rest-best-practices.md` — REST conventions reference
- `references/graphql-schema-design.md` — GraphQL schema patterns
